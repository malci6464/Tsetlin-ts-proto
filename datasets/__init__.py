"""Dataset utilities for the Tsetlin time-series prototypes."""

from .pamap2 import (
    PAMAP2_ARCHIVE_URL,
    PAMAP2_COLUMN_NAMES,
    Pamap2WindowDataset,
    binarize_windows,
    ensure_pamap2_dataset,
    list_subject_files,
    load_pamap2_dataframe,
    summarise_windows,
    DEFAULT_SENSOR_COLUMNS,
)

__all__ = [
    "PAMAP2_ARCHIVE_URL",
    "PAMAP2_COLUMN_NAMES",
    "Pamap2WindowDataset",
    "binarize_windows",
    "ensure_pamap2_dataset",
    "list_subject_files",
    "load_pamap2_dataframe",
    "summarise_windows",
    "DEFAULT_SENSOR_COLUMNS",
]
