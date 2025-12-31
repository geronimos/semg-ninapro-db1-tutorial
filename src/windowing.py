"""Sliding window logic for sEMG segmentation."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def iter_windows(
    emg: np.ndarray,
    restimulus: np.ndarray,
    rerepetition: np.ndarray,
    win_samples: int,
    hop_samples: int,
) -> Iterable[Tuple[np.ndarray, int, int]]:
    """Yield (window, label, repetition) tuples for one exercise."""
    T = emg.shape[0]
    if restimulus.shape[0] != T or rerepetition.shape[0] != T:
        raise ValueError("Label lengths do not match EMG length.")

    for t_end in range(win_samples - 1, T, hop_samples):
        t_start = t_end - win_samples + 1
        Xw = emg[t_start : t_end + 1, :]
        y = int(restimulus[t_end])
        rep = int(rerepetition[t_end])
        yield Xw, y, rep
