"""Utility functions for working with the PAMAP2 activity dataset.

The helpers in this module intentionally avoid downloading or processing the
full dataset at import time. Instead, they expose composable building blocks
for scripts that want to fetch the archive on demand, load selected subject
files, window the time-series signals, and turn the resulting statistics into
binary feature vectors consumable by a (MultiClass) Tsetlin Machine.

The implementation relies on :mod:`pandas` and :mod:`numpy` for data wrangling.
Both dependencies are ubiquitous in scientific Python environments, but they
are not bundled with this repository to keep the prototype lightweight. Scripts
using this module should declare them as optional extras when necessary.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd

PAMAP2_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/"
    "PAMAP2_Dataset.zip"
)
"""Location of the official PAMAP2 dataset archive."""

PAMAP2_COLUMN_NAMES: Tuple[str, ...] = (
    "timestamp",
    "activity_id",
    "heart_rate",
    "hand_temperature",
    "hand_acc_16g_x",
    "hand_acc_16g_y",
    "hand_acc_16g_z",
    "hand_acc_6g_x",
    "hand_acc_6g_y",
    "hand_acc_6g_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orientation_w",
    "hand_orientation_x",
    "hand_orientation_y",
    "hand_orientation_z",
    "chest_temperature",
    "chest_acc_16g_x",
    "chest_acc_16g_y",
    "chest_acc_16g_z",
    "chest_acc_6g_x",
    "chest_acc_6g_y",
    "chest_acc_6g_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orientation_w",
    "chest_orientation_x",
    "chest_orientation_y",
    "chest_orientation_z",
    "ankle_temperature",
    "ankle_acc_16g_x",
    "ankle_acc_16g_y",
    "ankle_acc_16g_z",
    "ankle_acc_6g_x",
    "ankle_acc_6g_y",
    "ankle_acc_6g_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orientation_w",
    "ankle_orientation_x",
    "ankle_orientation_y",
    "ankle_orientation_z",
)
"""Ordered feature names according to the dataset documentation."""

DEFAULT_SENSOR_COLUMNS: Tuple[str, ...] = (
    "heart_rate",
    "hand_acc_16g_x",
    "hand_acc_16g_y",
    "hand_acc_16g_z",
    "chest_acc_16g_x",
    "chest_acc_16g_y",
    "chest_acc_16g_z",
    "ankle_acc_16g_x",
    "ankle_acc_16g_y",
    "ankle_acc_16g_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
)
"""Lightweight default subset of channels for experimentation."""


@dataclass
class Pamap2WindowDataset:
    """Windowed dataset together with provenance metadata."""

    features: np.ndarray
    labels: np.ndarray
    metadata: Mapping[str, object]


def ensure_pamap2_dataset(
    destination_root: Path | str,
    *,
    url: str = PAMAP2_ARCHIVE_URL,
    archive_name: str = "PAMAP2_Dataset.zip",
) -> Path:
    """Download and extract the PAMAP2 dataset if it is missing.

    Parameters
    ----------
    destination_root:
        Directory where the archive will be stored and extracted.
    url:
        Download location. Defaults to the official UCI mirror.
    archive_name:
        Optional override for the archive file name.

    Returns
    -------
    Path
        Path pointing to the extracted dataset directory (``PAMAP2_Dataset``).
    """

    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    archive_path = destination_root / archive_name
    extract_path = destination_root / "PAMAP2_Dataset"

    if not archive_path.exists():
        with urllib.request.urlopen(url) as response, archive_path.open("wb") as handle:
            chunk_size = 1 << 20  # 1 MiB
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)

    if not extract_path.exists():
        with ZipFile(archive_path) as archive:
            archive.extractall(destination_root)

    return extract_path


def list_subject_files(dataset_root: Path | str) -> Mapping[int, Path]:
    """Return a mapping from subject identifier to ``.dat`` file path."""

    dataset_root = Path(dataset_root)
    subject_files: Dict[int, Path] = {}

    for folder in ("Protocol", "Optional"):
        candidate_dir = dataset_root / folder
        if not candidate_dir.exists():
            continue
        for path in sorted(candidate_dir.glob("subject*.dat")):
            stem = path.stem.replace("subject", "")
            if not stem.isdigit():
                continue
            subject_files[int(stem)] = path

    return subject_files


def _read_subject_file(path: Path, *, drop_idle: bool) -> pd.DataFrame:
    data_frame = pd.read_csv(
        path,
        sep=" ",
        header=None,
        names=PAMAP2_COLUMN_NAMES,
        na_values=["-1.000000", "-1.0"],
        engine="c",
        comment=None,
    )

    if drop_idle:
        data_frame = data_frame[data_frame["activity_id"] != 0]

    return data_frame.reset_index(drop=True)


def load_pamap2_dataframe(
    dataset_root: Path | str,
    *,
    subjects: Iterable[int] | None = None,
    drop_idle: bool = True,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load selected subjects into a combined :class:`pandas.DataFrame`.

    Parameters
    ----------
    dataset_root:
        Path returned by :func:`ensure_pamap2_dataset` (the folder containing
        ``Protocol`` and ``Optional`` recordings).
    subjects:
        Iterable of subject identifiers to load. When omitted, all available
        subject files are used.
    drop_idle:
        Whether to remove samples with the ``activity_id`` equal to ``0``.
    columns:
        Optional subset of columns to retain (``activity_id`` and ``timestamp``
        are always preserved).
    """

    dataset_root = Path(dataset_root)
    subject_files = list_subject_files(dataset_root)

    if not subject_files:
        raise FileNotFoundError(
            f"No PAMAP2 subject files found under {dataset_root}. Did you call "
            "ensure_pamap2_dataset first?"
        )

    if subjects is None:
        selected_subjects = sorted(subject_files)
    else:
        selected_subjects = sorted(int(identifier) for identifier in subjects)

    frames: List[pd.DataFrame] = []

    for subject_id in selected_subjects:
        try:
            path = subject_files[subject_id]
        except KeyError as exc:  # pragma: no cover - guard for typos
            raise KeyError(f"Subject {subject_id} is not present in the dataset") from exc

        frame = _read_subject_file(path, drop_idle=drop_idle)
        frame["subject_id"] = subject_id
        frames.append(frame)

    if not frames:
        raise ValueError("No subject data loaded from the provided identifiers")

    combined = pd.concat(frames, ignore_index=True)

    required = {"timestamp", "activity_id", "subject_id"}
    if columns is not None:
        missing = set(columns) - set(PAMAP2_COLUMN_NAMES)
        if missing:
            raise KeyError(f"Columns {sorted(missing)} are not present in PAMAP2")
        selected = list(required | set(columns))
        combined = combined[selected]
    else:
        combined = combined[list(required | set(PAMAP2_COLUMN_NAMES))]

    combined.sort_values(["subject_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Replace sentinel ``-1`` values with NaN so that later statistics ignore them.
    combined.replace(-1.0, np.nan, inplace=True)

    return combined


def _window_indices(length: int, window_size: int, step: int) -> Iterable[Tuple[int, int]]:
    if length < window_size:
        return []
    for start in range(0, length - window_size + 1, step):
        yield start, start + window_size


def summarise_windows(
    frame: pd.DataFrame,
    *,
    window_size: int,
    step: int,
    feature_columns: Sequence[str] | None = None,
) -> Pamap2WindowDataset:
    """Turn a subject-wise DataFrame into windowed summary statistics."""

    if feature_columns is None:
        feature_columns = DEFAULT_SENSOR_COLUMNS

    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise KeyError(f"Columns {sorted(missing)} are missing from the provided frame")

    grouped = frame.groupby("subject_id", sort=True)

    feature_vectors: List[List[float]] = []
    labels: List[int] = []
    provenance: List[Dict[str, object]] = []

    for subject_id, subject_frame in grouped:
        subject_values = subject_frame[feature_columns].to_numpy(dtype=float)
        label_values = subject_frame["activity_id"].to_numpy(dtype=float)
        timestamp_values = subject_frame["timestamp"].to_numpy(dtype=float)

        for start, stop in _window_indices(len(subject_frame), window_size, step):
            window = subject_values[start:stop]
            label_window = label_values[start:stop]
            timestamps = timestamp_values[start:stop]

            stats: List[float] = []
            for column_index in range(window.shape[1]):
                column = window[:, column_index]
                column = column[~np.isnan(column)]
                if column.size == 0:
                    mean = 0.0
                    std = 0.0
                else:
                    mean = float(np.mean(column))
                    std = float(np.std(column))
                stats.extend([mean, std])

            if np.all(np.isnan(label_window)):
                continue

            label_counts: MutableMapping[int, int] = {}
            for raw_label in label_window:
                if math.isnan(raw_label):
                    continue
                label = int(raw_label)
                label_counts[label] = label_counts.get(label, 0) + 1

            if not label_counts:
                continue

            dominant_label = max(label_counts.items(), key=lambda item: item[1])[0]

            feature_vectors.append(stats)
            labels.append(dominant_label)
            provenance.append(
                {
                    "subject_id": int(subject_id),
                    "start_timestamp": float(timestamps[0]),
                    "stop_timestamp": float(timestamps[-1]),
                }
            )

    if not feature_vectors:
        raise ValueError("No windows could be generated with the provided configuration")

    features_array = np.asarray(feature_vectors, dtype=float)
    labels_array = np.asarray(labels, dtype=int)

    metadata = {
        "window_size": int(window_size),
        "step": int(step),
        "feature_columns": list(feature_columns),
        "subjects": sorted({int(entry["subject_id"]) for entry in provenance}),
        "windows": len(feature_vectors),
        "provenance": provenance,
    }

    return Pamap2WindowDataset(features=features_array, labels=labels_array, metadata=metadata)


def binarize_windows(
    features: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert window statistics to binary features for a Tsetlin Machine."""

    if features.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_samples, n_features)")

    clean = np.nan_to_num(features, copy=True)

    if thresholds is None:
        thresholds = np.median(clean, axis=0)

    binary = (clean >= thresholds).astype(int)
    return binary, thresholds
