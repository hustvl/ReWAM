import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import hydra
import torch.distributed as dist
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "evaluation" / "robotwin" / "eval_robotwin_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _has_task_override(key: str) -> bool:
    for raw_override in HydraConfig.get().overrides.task:
        raw_key = raw_override.split("=", 1)[0].lstrip("+~")
        if raw_key == key:
            return True
    return False


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _write_summary_files(
    *,
    tasks: list[str],
    task_rates: dict[str, dict[str, float | None]],
    summary_csv: Path,
    summary_json: Path,
) -> None:
    clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
    random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
        for task in tasks:
            writer.writerow(
                [
                    task,
                    task_rates[task]["clean"],
                    task_rates[task]["random"],
                ]
            )
        writer.writerow(["__overall__", clean_mean, random_mean])

    payload = {
        "per_task": [
            {
                "task_name": task,
                "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                "random_success_rate": _to_jsonable(task_rates[task]["random"]),
            }
            for task in tasks
        ],
        "overall": {
            "clean_mean_success_rate": _to_jsonable(clean_mean),
            "random_mean_success_rate": _to_jsonable(random_mean),
        },
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_failed_records(failed_tasks_file: Path, failed_records: list[dict[str, Any]]) -> None:
    with failed_tasks_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for rec in failed_records:
            row = [
                str(rec["task_name"]),
                str(rec["phase"]),
                f"gpu={rec['gpu_id']}",
                f"return_code={rec['return_code']}",
                f"reason={rec['reason']}",
            ]
            if rec.get("log_file"):
                row.append(f"log={rec['log_file']}")
            if rec.get("result_file"):
                row.append(f"result={rec['result_file']}")
            writer.writerow(row)


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _make_worker_log_file(run_output_dir: Path, task_name: str, phase: str, gpu_id: int) -> Path:
    task_part = _safe_filename(task_name)
    phase_part = _safe_filename(phase)
    return run_output_dir / (f"worker_{task_part}_{phase_part}_gpu{gpu_id}_{_timestamp_for_filename()}.log")


def _write_abort_signal(
    *,
    abort_file: Path,
    node_rank: int,
    failure_message: str,
) -> None:
    if abort_file.exists():
        return
    payload = {
        "node_rank": node_rank,
        "failure_message": failure_message,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp_path = abort_file.with_name(f".{abort_file.name}.{node_rank}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    try:
        os.replace(tmp_path, abort_file)
    except FileExistsError:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _read_abort_signal(abort_file: Path) -> dict[str, Any] | None:
    if not abort_file.exists():
        return None
    try:
        return json.loads(abort_file.read_text(encoding="utf-8"))
    except Exception:
        return {
            "node_rank": "unknown",
            "failure_message": f"failed to parse abort signal: {abort_file}",
        }


def _cfg_or_env(cfg: DictConfig, key: str, env_key: str, default: Any) -> Any:
    value = cfg.MULTIRUN.get(key)
    if value is not None:
        return value
    return os.environ.get(env_key, default)


def _latest_worker_log(run_output_dir: Path, task_name: str, phase: str) -> Path | None:
    if phase not in {"clean", "random"} or not run_output_dir.exists():
        return None
    prefix = f"eval_{task_name}_{phase}_"
    candidates = [
        path
        for path in run_output_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.suffix == ".log"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _sync_multinode_status(
    *,
    master_addr: str,
    master_port: int,
    num_nodes: int,
    node_rank: int,
    timeout_s: int,
    has_failure: bool,
    failure_message: str,
) -> list[dict[str, Any]]:
    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=num_nodes,
        is_master=(node_rank == 0),
        timeout=timedelta(seconds=timeout_s),
    )
    key = f"robotwin_eval_status::{node_rank}"
    store.set(
        key,
        json.dumps(
            {
                "node_rank": node_rank,
                "has_failure": bool(has_failure),
                "failure_message": str(failure_message),
            },
            ensure_ascii=True,
        ),
    )
    statuses = []
    for rank in range(num_nodes):
        payload = store.get(f"robotwin_eval_status::{rank}").decode("utf-8")
        statuses.append(json.loads(payload))

    # TCPStore's master lives in the rank-0 process. Keep rank 0 alive until
    # every rank has finished reading the shared statuses, otherwise slower
    # ranks can see connection-reset errors while still inside store.get().
    read_ack_key = f"robotwin_eval_status_read_ack::{node_rank}"
    store.set(read_ack_key, "1")
    if node_rank == 0:
        for rank in range(num_nodes):
            store.get(f"robotwin_eval_status_read_ack::{rank}")
        store.set("robotwin_eval_status_release", "1")

    store.get("robotwin_eval_status_release")

    release_ack_key = f"robotwin_eval_status_release_ack::{node_rank}"
    store.set(release_ack_key, "1")
    if node_rank == 0:
        for rank in range(num_nodes):
            store.get(f"robotwin_eval_status_release_ack::{rank}")

    return statuses


@dataclass(frozen=True)
class MultiNodeEvalCoordinator:
    run_output_dir: Path
    num_nodes: int
    node_rank: int
    master_addr: str
    master_port: int
    barrier_timeout_s: int

    @property
    def enabled(self) -> bool:
        return self.num_nodes > 1

    @property
    def abort_file(self) -> Path:
        return self.run_output_dir / "multinode_abort.json"

    def write_abort(self, failure_message: str) -> None:
        if self.enabled:
            _write_abort_signal(
                abort_file=self.abort_file,
                node_rank=self.node_rank,
                failure_message=failure_message,
            )

    def read_abort(self) -> dict[str, Any] | None:
        return _read_abort_signal(self.abort_file)

    def remote_abort_message(self) -> str | None:
        if not self.enabled:
            return None
        abort_payload = self.read_abort()
        if abort_payload is None:
            return None
        origin_rank = abort_payload.get("node_rank", "unknown")
        origin_failure = abort_payload.get("failure_message", "")
        return f"aborted due to remote node failure: node_rank={origin_rank}, " f"reason={origin_failure}"

    def sync_statuses(self, *, has_failure: bool, failure_message: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return [
                {
                    "node_rank": self.node_rank,
                    "has_failure": has_failure,
                    "failure_message": failure_message,
                }
            ]
        return _sync_multinode_status(
            master_addr=self.master_addr,
            master_port=self.master_port,
            num_nodes=self.num_nodes,
            node_rank=self.node_rank,
            timeout_s=self.barrier_timeout_s,
            has_failure=has_failure,
            failure_message=failure_message,
        )


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]
    log_file: Path


@dataclass
class EvalManager:
    ckpt_path: Path
    output_dir: Path
    run_output_dir: Path
    manager_log: Path
    tasks: list[str]
    all_tasks: list[str]
    extra_overrides: list[str]
    max_tasks_per_gpu: int
    multinode: MultiNodeEvalCoordinator
    failed_tasks_file: Path
    summary_csv: Path
    summary_json: Path
    final_failed_tasks_file: Path
    final_summary_csv: Path
    final_summary_json: Path
    task_rates: dict[str, dict[str, float | None]]
    failed_records: list[dict[str, Any]]
    pending_tasks: deque
    running_states: list[RunningState]
    has_failure: bool = False
    failure_message: str = ""

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(self, *, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = self.phase_to_task_config[phase]
        cmd = [
            sys.executable,
            "-u",
            str(SINGLE_ENTRY),
            f"ckpt={str(self.ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(self.output_dir)}",
        ]
        cmd.extend(self.extra_overrides)
        return cmd

    def launch_phase(self, task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = self.build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        worker_log_file = _make_worker_log_file(
            run_output_dir=self.run_output_dir,
            task_name=task_name,
            phase=phase,
            gpu_id=gpu_id,
        )
        self.log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"log={worker_log_file} cmd={' '.join(cmd)}"
        )
        with worker_log_file.open("w", encoding="utf-8") as worker_log_f:
            worker_log_f.write(f"[manager] cwd={PROJECT_ROOT}\n")
            worker_log_f.write(f"[manager] cmd={' '.join(cmd)}\n")
            worker_log_f.flush()
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=worker_log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
            log_file=worker_log_file,
        )

    def terminate_all_running(self) -> None:
        for state in list(self.running_states):
            if state.process.poll() is not None:
                continue
            self.log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(self.running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                self.log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def apply_remote_abort_if_any(self) -> bool:
        if self.has_failure:
            return False
        remote_message = self.multinode.remote_abort_message()
        if remote_message is None:
            return False
        self.has_failure = True
        self.failure_message = remote_message
        self.log(remote_message)
        self.terminate_all_running()
        self.running_states.clear()
        return True

    def gpu_running_count(self, gpu_id: int) -> int:
        count = 0
        for state in self.running_states:
            if state.gpu_id == gpu_id and state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(self, gpu_id: int) -> None:
        while len(self.pending_tasks) > 0 and self.gpu_running_count(gpu_id) < self.max_tasks_per_gpu:
            task_name = self.pending_tasks.popleft()
            self.running_states.append(self.launch_phase(task_name=task_name, gpu_id=gpu_id, phase="clean"))

    def fail(self, failure_message: str, failed_record: dict[str, Any]) -> None:
        self.has_failure = True
        self.failure_message = failure_message
        self.failed_records.append(failed_record)
        self.log(failure_message)
        self.multinode.write_abort(failure_message)
        self.terminate_all_running()
        self.running_states.clear()

    def write_outputs(self) -> None:
        _write_summary_files(
            tasks=self.tasks,
            task_rates=self.task_rates,
            summary_csv=self.summary_csv,
            summary_json=self.summary_json,
        )
        _write_failed_records(self.failed_tasks_file, self.failed_records)

    def write_final_outputs(self) -> None:
        final_task_rates: dict[str, dict[str, float | None]] = {}
        final_failed_records: list[dict[str, Any]] = []

        for task in self.all_tasks:
            final_task_rates[task] = {"clean": None, "random": None}
            for phase in ("clean", "random"):
                result_file = self.run_output_dir / task / _phase_result_filename(phase)
                try:
                    final_task_rates[task][phase] = _parse_success_rate(result_file)
                except Exception as exc:
                    final_failed_records.append(
                        {
                            "task_name": task,
                            "phase": phase,
                            "gpu_id": -1,
                            "return_code": -1,
                            "reason": f"final_result_parse_failed:{repr(exc)}",
                        }
                    )

        _write_summary_files(
            tasks=self.all_tasks,
            task_rates=final_task_rates,
            summary_csv=self.final_summary_csv,
            summary_json=self.final_summary_json,
        )
        _write_failed_records(self.final_failed_tasks_file, final_failed_records)

        if final_failed_records:
            raise RuntimeError(f"final summary has {len(final_failed_records)} missing or invalid results")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    num_nodes = int(_cfg_or_env(cfg, "num_nodes", "NNODES", 1))
    node_rank = int(_cfg_or_env(cfg, "node_rank", "NODE_RANK", 0))
    if num_nodes <= 0:
        raise ValueError("`MULTIRUN.num_nodes` must be > 0.")
    if node_rank < 0 or node_rank >= num_nodes:
        raise ValueError(f"`MULTIRUN.node_rank` must be in [0, {num_nodes}), got {node_rank}.")
    if num_nodes > 1 and not _has_task_override("EVALUATION.output_dir"):
        raise ValueError(
            "Multi-node RoboTwin eval requires an explicit shared `EVALUATION.output_dir`. "
            "Pass the same value on every node so all workers write into one run directory."
        )
    master_addr = str(_cfg_or_env(cfg, "master_addr", "MASTER_ADDR", "127.0.0.1"))
    master_port = int(_cfg_or_env(cfg, "master_port", "EVAL_MASTER_PORT", 29523))
    barrier_timeout_s = int(_cfg_or_env(cfg, "barrier_timeout_s", "EVAL_BARRIER_TIMEOUT", 3600))
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_output_dir = output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / f"manager_node_{node_rank}.log"
    failed_tasks_file = run_output_dir / f"failed_tasks_node_{node_rank}.txt"
    summary_csv = run_output_dir / f"summary_node_{node_rank}.csv"
    summary_json = run_output_dir / f"summary_node_{node_rank}.json"
    final_failed_tasks_file = run_output_dir / "failed_tasks.txt"
    final_summary_csv = run_output_dir / "summary.csv"
    final_summary_json = run_output_dir / "summary.json"
    multinode = MultiNodeEvalCoordinator(
        run_output_dir=run_output_dir,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        barrier_timeout_s=barrier_timeout_s,
    )

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        all_tasks = _load_all_tasks()
    else:
        all_tasks = [str(task_name_cfg)]
    tasks = [task for idx, task in enumerate(all_tasks) if idx % num_nodes == node_rank]

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {task: {"clean": None, "random": None} for task in tasks}
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []
    manager = EvalManager(
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        run_output_dir=run_output_dir,
        manager_log=manager_log,
        tasks=tasks,
        all_tasks=all_tasks,
        extra_overrides=extra_overrides,
        max_tasks_per_gpu=max_tasks_per_gpu,
        multinode=multinode,
        failed_tasks_file=failed_tasks_file,
        summary_csv=summary_csv,
        summary_json=summary_json,
        final_failed_tasks_file=final_failed_tasks_file,
        final_summary_csv=final_summary_csv,
        final_summary_json=final_summary_json,
        task_rates=task_rates,
        failed_records=failed_records,
        pending_tasks=pending_tasks,
        running_states=running_states,
    )

    manager.log(
        f"manager start node_rank={node_rank}/{num_nodes} local_tasks={len(tasks)} "
        f"all_tasks={len(all_tasks)} gpu_ids={gpu_ids} max_tasks_per_gpu={max_tasks_per_gpu} "
        f"output_dir={run_output_dir}"
    )

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        manager.try_launch_pending(gpu_id)

    while len(manager.running_states) > 0:
        progressed = False
        for state in list(manager.running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            manager.running_states.remove(state)

            if return_code != 0:
                worker_log = _latest_worker_log(run_output_dir, state.task_name, state.phase)
                log_path = worker_log or state.log_file
                worker_log_text = f", log={log_path}"
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}{worker_log_text}"
                )
                manager.fail(
                    failure_message,
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                        "log_file": str(log_path),
                    },
                )
                break

            result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                worker_log = _latest_worker_log(run_output_dir, state.task_name, state.phase)
                log_path = worker_log or state.log_file
                worker_log_text = f", log={log_path}"
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}{worker_log_text}"
                )
                manager.fail(
                    failure_message,
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                        "log_file": str(log_path),
                        "result_file": str(result_file),
                    },
                )
                break

            manager.task_rates[state.task_name][state.phase] = success_rate
            manager.log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )

            if state.phase == "clean":
                manager.running_states.append(
                    manager.launch_phase(
                        task_name=state.task_name,
                        gpu_id=gpu_id,
                        phase="random",
                    )
                )
                continue

            manager.try_launch_pending(gpu_id)

        if manager.has_failure:
            break
        if manager.apply_remote_abort_if_any():
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started tasks when failure happened.
    if manager.has_failure:
        for task_name in manager.pending_tasks:
            manager.failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    manager.write_outputs()
    manager.log(f"node summary saved: {summary_csv} and {summary_json}")

    statuses = [
        {
            "node_rank": node_rank,
            "has_failure": manager.has_failure,
            "failure_message": manager.failure_message,
        }
    ]
    if multinode.enabled:
        manager.log(
            f"waiting for multinode eval barrier addr={master_addr} "
            f"port={master_port} timeout_s={barrier_timeout_s}"
        )
        try:
            statuses = multinode.sync_statuses(
                has_failure=manager.has_failure,
                failure_message=manager.failure_message,
            )
        except Exception as exc:
            abort_payload = multinode.read_abort()
            abort_text = ""
            if abort_payload is not None:
                abort_text = (
                    f"; abort_signal=node_rank={abort_payload.get('node_rank')}, "
                    f"reason={abort_payload.get('failure_message')}"
                )
            local_status = (
                f"local node {node_rank} status: has_failure={manager.has_failure}, "
                f"failure_message={manager.failure_message or '<none>'}"
            )
            raise RuntimeError(
                "multinode eval barrier failed before all nodes reported status; "
                f"{local_status}{abort_text}. Original error: {repr(exc)}"
            ) from exc
        manager.log("multinode eval barrier passed")

    failed_statuses = [status for status in statuses if status.get("has_failure")]
    if node_rank == 0:
        if failed_statuses:
            with final_failed_tasks_file.open("w", encoding="utf-8") as f:
                for status in failed_statuses:
                    f.write(
                        f"node_rank={status.get('node_rank')}," f"reason={status.get('failure_message')}\n"
                    )
        else:
            manager.write_final_outputs()
            manager.log(f"final summary saved: {final_summary_csv} and {final_summary_json}")

    if failed_statuses:
        raise RuntimeError(
            "; ".join(
                f"node {status.get('node_rank')}: {status.get('failure_message')}"
                for status in failed_statuses
            )
        )

    manager.log("manager finished successfully")


if __name__ == "__main__":
    main()
