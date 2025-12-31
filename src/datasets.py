"""Dataset construction helpers for NinaPro DB1."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.features import extract_time_domain_features
from src.windowing import iter_windows


def _next_repetition_labels(rerepetition: np.ndarray) -> np.ndarray:
    """Map rest samples to the next non-zero repetition."""
    next_rep = np.zeros_like(rerepetition, dtype=np.int64)
    next_nonzero = 0
    for idx in range(rerepetition.shape[0] - 1, -1, -1):
        rep = int(rerepetition[idx])
        if rep != 0:
            next_nonzero = rep
            next_rep[idx] = rep
        else:
            next_rep[idx] = next_nonzero
    return next_rep


def _build_movement_id_maps(exercises: Dict[int, Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """Create per-exercise lookup tables to remap movement IDs to global IDs."""
    offset = 0
    maps: Dict[int, np.ndarray] = {}
    for ex_id in sorted(exercises.keys()):
        restimulus = np.asarray(exercises[ex_id]["restimulus"]).astype(int)
        labels = np.unique(restimulus)
        active = labels[labels > 0]
        active.sort()

        max_label = int(active.max()) if active.size else 0
        lookup = np.full(max_label + 1, -1, dtype=np.int64)
        lookup[0] = 0
        for idx, label in enumerate(active, start=offset + 1):
            lookup[int(label)] = idx
        if np.any(lookup[labels] < 0):
            raise ValueError(f"Missing movement mapping for exercise {ex_id}.")

        maps[ex_id] = lookup
        offset += active.size
    return maps


def _remap_restimulus(restimulus: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """Map per-exercise restimulus labels to global movement IDs."""
    restimulus = np.asarray(restimulus).astype(int)
    if restimulus.size == 0:
        return restimulus.astype(np.int64)
    max_label = int(restimulus.max())
    if max_label >= lookup.shape[0]:
        raise ValueError("Restimulus contains labels outside the lookup table.")
    mapped = lookup[restimulus]
    if np.any(mapped < 0):
        raise ValueError("Restimulus contains unmapped labels.")
    return mapped


def build_subject_dataset(
    subject_id: int,
    db: Dict[int, Dict[str, Any]],
    win_samples: int,
    hop_samples: int,
    train_reps: set,
    test_reps: set,
) -> Dict[str, np.ndarray]:
    """Build a within-subject dataset using repetition-based split.

    Movement IDs are remapped across exercises to a global contiguous range
    (rest=0).
    """
    if subject_id not in db:
        raise KeyError(f"Subject {subject_id} not found in DB.")

    Xtr, ytr = [], []
    Xte, yte = [], []

    exercises = db[subject_id]["exercises"]
    movement_maps = _build_movement_id_maps(exercises)
    for ex_id in sorted(exercises.keys()):
        rec = exercises[ex_id]
        emg = rec["emg"]
        rest = _remap_restimulus(rec["restimulus"], movement_maps[ex_id])
        rerep = rec["rerepetition"]
        split_reps = _next_repetition_labels(rerep)

        t_end = win_samples - 1
        for Xw, y, _ in iter_windows(emg, rest, rerep, win_samples, hop_samples):
            split_rep = int(split_reps[t_end])
            if split_rep in test_reps:
                feats = extract_time_domain_features(Xw)
                Xte.append(feats)
                yte.append(y)
            elif split_rep in train_reps:
                feats = extract_time_domain_features(Xw)
                Xtr.append(feats)
                ytr.append(y)
            t_end += hop_samples

    X_train = np.stack(Xtr, axis=0) if Xtr else np.empty((0, 0), dtype=np.float32)
    y_train = np.asarray(ytr, dtype=np.int64)
    X_test = np.stack(Xte, axis=0) if Xte else np.empty((0, 0), dtype=np.float32)
    y_test = np.asarray(yte, dtype=np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
