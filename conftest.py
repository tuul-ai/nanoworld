"""Global test config: every test runs on CPU, bit-deterministic, fixed seed.

MPS/CUDA runs are seed-logged but not bit-reproducible (see README determinism
policy); tests are the tier that must never flake.
"""
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NANOWORLD_DEVICE"] = "cpu"

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
