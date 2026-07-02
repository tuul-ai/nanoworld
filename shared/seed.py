"""Seeding + device selection. Every train.py calls seed_all() first and logs the seed."""
import os
import random

import numpy as np
import torch


def seed_all(seed: int) -> torch.Generator:
    """Seed python, numpy, torch (+mps/cuda when available); return a Generator for samplers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def pick_device() -> torch.device:
    """mps -> cuda -> cpu; NANOWORLD_DEVICE env var overrides (tests force cpu)."""
    override = os.environ.get("NANOWORLD_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
