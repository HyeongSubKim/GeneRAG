# -*- coding: utf-8 -*-
"""
Small helpers shared by examples and notebooks.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 0) -> None:
    """Make NumPy, Python, and (if available) PyTorch deterministic-ish.

    Useful for reproducible quickstart runs; does not guarantee
    bit-exact reproducibility of CUDA kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def auto_device(prefer: str = "cuda") -> str:
    """Return a usable device string.

    Returns ``prefer`` when CUDA is available; otherwise falls back to
    ``'cpu'``. ``prefer`` may already be ``'cuda:0'``, ``'cuda:3'``, etc.
    """
    try:
        import torch
        if prefer.startswith("cuda") and torch.cuda.is_available():
            return prefer
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def ensure_dir(path: str) -> str:
    """``mkdir -p`` and return ``path`` for chaining."""
    os.makedirs(path, exist_ok=True)
    return path
