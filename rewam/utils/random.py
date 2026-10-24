import os
import random

import numpy as np
import torch


def _resolve_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0"))))


def set_global_seed(seed: int) -> None:
    """Seed training noise per rank; model initialization uses a shared seed."""
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("Seed must be within the np.uint32 bounds.")

    # Process-specific seeding: offset by global rank so each process gets a different seed
    global_rank = _resolve_global_rank()
    process_seed = seed + global_rank

    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def worker_init_function(worker_id: int) -> None:
    """Preserve the existing Lightning-derived NumPy SeedSequence worker streams.

    Each worker mixes its DataLoader seed, worker ID and process rank, then seeds
    NumPy, Torch and Python separately.
    """
    process_seed = torch.initial_seed()
    global_rank = _resolve_global_rank()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq, _ = seed_seq.spawn(3)

    # Torch takes 64 bits; Python uses the two-word 128-bit seed below.
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)
