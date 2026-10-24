> 2026-10-19 后续变更：本页为历史记录。独立版已经删除内置 2.x/3.0 读取器，
> 改为官方 LeRobot 0.4.4 加本地只读格式适配；当前实现见 [开发记录](../../LEROBOT_DEPENDENCY_REFACTOR.md)。

# FastWAM 最新 8 个提交审查与整合（2026-10-14）

审查对象为 [FastWAM 上游](https://github.com/yuantianyuan01/FastWAM) 的 main，
固定到 `7faa71108368fbb3b6885649f112af607427a2d4`。
最新 8 个提交的范围为 `45d8e14..7faa711`，提交日期为 2026-08-13 至 2026-08-20。
本次逐项移植适用的实现，没有直接 merge 或 cherry-pick，以保留当前 ReWAM 架构。

## 逐提交结论

| 提交（由新到旧） | 上游内容 | 本次处理 |
| --- | --- | --- |
| [7faa711](https://github.com/yuantianyuan01/FastWAM/commit/7faa71108368fbb3b6885649f112af607427a2d4) | IDM 动作推理不再额外解码未来视频，仅在请求视频时解码 | 当前 ReWAM 动作推理已与 RGB 诊断分离，无需引入 IDM 或 VAE。 |
| [9af5462](https://github.com/yuantianyuan01/FastWAM/commit/9af5462da5a6ac345abd6872ca89dee35ff011f5) | Optional IDM 发布说明与实验结果 | 不适用于保留的 ReWAM 模型，不引入该模型或配置。 |
| [fde8ea4](https://github.com/yuantianyuan01/FastWAM/commit/fde8ea4694841d91c0359687737251f42b69a170) | Joint 评测入口接收并忽略编译选项，避免调用参数不兼容 | ReWAM 只向动作推理传入编译选项，未来视频诊断沿用原入口，避免添加被忽略的参数。 |
| [8e6e4cb](https://github.com/yuantianyuan01/FastWAM/commit/8e6e4cb33bf0152c7cc587bedcdfc85bc6398e68) | 加速、LeRobot 3.0 和新模型的使用说明，以及 LIBERO 3.0 配置调整 | 将适用能力写入中文 ReWAM 文档，不复制 LIBERO 配置或上游性能结论。 |
| [0050a44](https://github.com/yuantianyuan01/FastWAM/commit/0050a44dfce439e5c2345870fbfd35c76d85a463) | 动作调度器训练与推理 shift 从 5.0 改为 1.0 | 保留 ReWAM 的 5.0。上游 FastWAM 的实验结果不足以支持改动 ReWAM 的训练或历史权重评测设置。 |
| [11409b6](https://github.com/yuantianyuan01/FastWAM/commit/11409b6571d64a83302d203ae6a35bc0b0e9eda3) | LIBERO 常驻 worker、动态任务队列、故障 GPU 隔离及结果恢复 | 不引入 LIBERO。RoboTwin 的 clean/random 子进程与多节点管理流程不同，不能直接套用该常驻仿真器协议；本次不改动其调度行为。 |
| [4d46f0a](https://github.com/yuantianyuan01/FastWAM/commit/4d46f0a04b5bddd2d18fe87e4e353a03b024e37d) | LeRobot 3.0 的分块 parquet、共享视频、较少图像帧读取，以及依赖调整 | 引入本地只读后端，由元数据自动选择 2.x/3.0；保留现有单一处理器与时序。未跟随修改整套依赖版本。 |
| [8afd20f](https://github.com/yuantianyuan01/FastWAM/commit/8afd20fd01b965cea67c3824556ebbb26f970e9b) | 编译推理与训练、RoPE 设备驻留、VAE 批处理、在线文本、数据列兼容修复、Optional IDM | 整合 RoPE 驻留、可选编译动作推理、在线文本和数据列修复。ReWAM 已有稀疏图像读取与世界 K/V 缓存。未引入 VAE、IDM 或禁用梯度检查点的训练替换路径。 |

## 整合后的实现

### 推理

动作与世界专家的 RoPE 频率表随模型迁移到目标设备，保持原来的 complex128 精度，
避免每个去噪步骤从 CPU 传输频率表。频率表不进入 checkpoint。

动作文本投影、文本掩码和动作 RoPE 每次请求准备一次；每个去噪步骤只更新动作 token 和时间调制。
本体状态仍按原逻辑拼入当前请求的条件，不跨请求复用文本条件或世界状态缓存。

`EVALUATION.compile_action_infer=true` 可编译动作去噪核心。
编译直接复用现有 MoT，无第二套网络实现，也不注册编译包装模型到 state-dict。
默认关闭，首次编译有启动开销。编译结果用于调度器更新后复制到独立张量，
避免后续 CUDA Graph 重放覆盖仍被使用的输出。世界 K/V 由原有非编译路径生成。
未来 RGB 诊断仍使用原有 `infer_joint()`，不传入动作编译参数。

```bash
python evaluation/robotwin/run_robotwin_manager.py \
  task=robotwin_rewam ckpt=./runs/RUN/checkpoints/weights/step_STEP.pt \
  EVALUATION.dataset_stats_path=./runs/RUN/dataset_stats.json \
  EVALUATION.compile_action_infer=true
```

### 数据与文本

`BaseLerobotDataset` 依据各数据目录 `meta/info.json` 中的 `codebase_version` 自动选择读取后端。
支持原有 LeRobot 2.0/2.1，以及新的 3.0 布局；一次数据组合使用同一种存储布局。
无需新增 RoboTwin 数据集子类、处理器子类或重复 Hydra 配置。

3.0 后端支持分块 parquet、共享视频中的 episode 起始时间偏移、episode 子集、边界填充、
数据集拼接及完整 episode 读取。episode 与 task 按实际 ID 查询，
不假设元数据的行号等于 ID；数据按 episode/frame 排序，确保采样不跨越 episode。
元数据读取时排除 `stats/` 列，归一化仍由现有 ReWAM 流程计算或加载原统计文件。

现有 2.x 后端移除了无效的全量 timestamp/episode 列加载，并兼容可迭代的数据列对象。
保留原有挂载文件系统加载重试和逐 episode 统计流程。
新的 3.0 后端已在现有 `datasets==3.6.0` 上验证，无需为此升级 numpy、DeepSpeed 等依赖。
上游调整后的整套依赖组合尚未验证。

最终时序仍为 `[-8,0,8,...,64]`，十帧形成五个帧对，预测 64 步动作。
已有稀疏图像读取覆盖了上游预采样优化，未新增 stride 或 presample 开关。

默认读取缓存文本。需要在线 UMT5 时，可以显式设置：

```bash
bash scripts/train.sh 8 task=robotwin_rewam \
  data.train.text_embed_cache_dir=null model.load_text_encoder=true
```

`text_embed_cache_dir` 有值时读取缓存，为 `null` 或空字符串时使用在线文本编码。
验证集跟随训练集的文本缓存设置。在线路径返回原有 prompt，模型侧执行原有冻结文本编码；
缓存路径保留补零和全有效文本掩码的既有行为。
`context` 与 `context_mask` 必须同时提供，避免缺失其中一项时静默切换文本来源。

### 保留原有训练行为的原因

上游训练编译实现拒绝梯度检查点，并在 DeepSpeed 初始化前执行一次真实训练损失和反向传播作为预热。
ReWAM 的损失还涉及滞后归一化统计、重建诊断及 AGRS 梯度传递。
直接照搬预热会更新统计状态，训练编译需要单独设计状态隔离并验证 GPU/分布式行为。
本次保留原训练路径及梯度检查点能力，没有添加未经验证的训练加速开关。

## 权重与现有脚本

本轮未改变权重键、张量形状、参数顺序、默认调度器、采样时间线或统计文件格式。
尚未转换任何已训练权重；之前重构记录中的名称迁移仍然需要后续确认与执行。
开启编译后，GPU 数值可能存在编译后端引起的浮点差异，需用迁移后的真实权重单独验证。

`fastwam_utils/run_.sh` 仍使用原有 2.x 数据、缓存文本、默认非编译推理与未来视频诊断，
无需改动。新增能力通过现有 Hydra 配置入口选择。本轮未修改 HWM、插件版或独立 StarVLA。

## 验证

独立版完整 CPU 测试 34 项通过，其中本轮新增 15 项，覆盖：

- 动作条件每个请求只投影一次，结果与旧去噪循环逐项相等。
- 使用 `torch.compile(backend="eager", fullgraph=True)` 捕获完整动作图，
  不同请求条件及动作长度下结果相等；编译后权重键不变。
- RoPE 随设备迁移，低精度权重转换后频率值与复数精度保持原样。
- 2.x 与 3.0 原生动作、状态、填充掩码、图像和归一化统计相等。
- 3.0 非连续 episode ID、乱序 task 元数据、子集、拼接、训练/验证划分及视频时间偏移。
- 实际生成的微型 MP4 经 PyAV 解码，检查共享视频的 episode 偏移。
- 完整 RobotVideoDataset/处理器路径保留十帧和 64 步动作。
- 在线与缓存文本路径、部分文本缓存字段缺失时的明确报错，以及 RoboTwin 入口的编译配置传递。
- 2.x 读取兼容可迭代的非 list 数据列。

```bash
# 在 rewam_fastwam 目录，已安装项目依赖的环境中执行
PYTHONPATH=. python -m pytest -q tests
```

本地测试使用现有 starVLA 的 Python 3.10/PyTorch 2.8 环境，
并在测试进程中复用现有 fastwam 环境的 datasets 3.6.0 安装，没有重建环境。
另用本轮开始前的源码快照比较有、无重建探针两种小模型：全部权重、参数顺序、训练损失、
所有参数梯度和固定种子动作推理完全相等。仅使用人工初始化权重。

修改代码通过 Black、针对修改模块的 Ruff 和 `git diff --check`。
尚未验证真实 UMT5/DINO 权重、生产数据、GPU Inductor/CUDA Graph 性能、分布式训练或仿真器。
CPU 的 eager 编译检查证明图可完整捕获，不代表 GPU 加速倍数或显存收益。

## 本轮修改文件

源码与配置（相对于本轮开始前的快照）：

- `rewam/models/action_dit.py`
- `rewam/models/wan22/wan_video_dit.py`
- `rewam/models/rewam.py`
- `rewam/datasets/lerobot/base_lerobot_dataset.py`
- `rewam/datasets/lerobot/robot_video_dataset.py`
- `rewam/datasets/lerobot/lerobot/lerobot_dataset.py`
- `rewam/datasets/lerobot/lerobot_v3.py`（新增，保留上游来源与许可证头）
- `evaluation/robotwin/rewam_policy/deploy_policy.py`
- `configs/data/robotwin.yaml`
- `configs/sim_robotwin.yaml`
- `tests/test_upstream_inference.py`（新增）
- `tests/test_upstream_data.py`（新增）

另更新 `README.md`、`docs/checkpoint_compatibility.md` 和工作区的 `VALIDATION.md`，
并新增本整合记录。原有其他重构变更保留。
