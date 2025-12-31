"""Feature extraction utilities for sEMG windows."""

from __future__ import annotations

import numpy as np


def mean_absolute_value(window: np.ndarray) -> np.ndarray:
    """Mean Absolute Value (MAV) feature per channel."""
    return np.mean(np.abs(window), axis=0)


def waveform_length(window: np.ndarray) -> np.ndarray:
    """Waveform Length (WL) feature per channel."""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def variance(window: np.ndarray) -> np.ndarray:
    """Variance (VAR) feature per channel."""
    return np.var(window, axis=0, ddof=1)


def extract_time_domain_features(window: np.ndarray) -> np.ndarray:
    """Compute MAV, VAR, and WL per channel and concatenate."""
    mav = mean_absolute_value(window)
    var = variance(window)
    wl = waveform_length(window)
    return np.concatenate([mav, var, wl]).astype(np.float32)
