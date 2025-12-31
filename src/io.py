"""Data loading and parsing utilities for NinaPro DB1."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.io import loadmat


def load_db1_a1_mat(mat_path: Path) -> Dict[str, Any]:
    """Load one NinaPro DB1 A1 .mat file and return a dict with key arrays."""
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    subject = int(np.atleast_1d(mat["subject"])[0])
    exercise = int(np.atleast_1d(mat["exercise"])[0])

    emg = np.asarray(mat["emg"])
    glove = np.asarray(mat["glove"])
    stimulus = np.asarray(mat["stimulus"]).astype(int)
    restimulus = np.asarray(mat["restimulus"]).astype(int)
    repetition = np.asarray(mat["repetition"]).astype(int)
    rerepetition = np.asarray(mat["rerepetition"]).astype(int)

    return {
        "subject_id": subject,
        "exercise_id": exercise,
        "emg": emg,
        "glove": glove,
        "stimulus": stimulus,
        "restimulus": restimulus,
        "repetition": repetition,
        "rerepetition": rerepetition,
        "path": mat_path,
    }


def build_db1_a1_database(data_root: Path, acq_setup: str = "A1") -> Dict[int, Dict[str, Any]]:
    """Build a nested dictionary for DB1 A1 across subjects and exercises."""
    db: Dict[int, Dict[str, Any]] = {}

    for subj_dir in sorted(data_root.glob("s*")):
        if not subj_dir.is_dir():
            continue

        try:
            subject_id = int(subj_dir.name.lstrip("s"))
        except ValueError:
            continue

        mat_files = sorted(subj_dir.glob(f"S{subject_id}_{acq_setup}_E*.mat"))
        if not mat_files:
            continue

        db[subject_id] = {
            "meta": {
                "subject_id": subject_id,
            },
            "exercises": {},
        }

        for mat_path in mat_files:
            rec = load_db1_a1_mat(mat_path)
            ex_id = rec["exercise_id"]
            db[subject_id]["exercises"][ex_id] = rec

    return db


def save_db_pickle(db: Dict[int, Dict[str, Any]], out_path: Path) -> Path:
    """Persist the database dictionary to disk as a pickle."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path
