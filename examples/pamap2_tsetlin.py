"""Train a (surrogate) Tsetlin Machine on PAMAP2 activity windows.

The script demonstrates how to combine the runtime download helpers from
``datasets.pamap2`` with the perceptron-based ``MultiClassTsetlinMachine``
shim bundled in this repository. It is intentionally lightweight so that the
end-to-end workflow can execute on a laptop while still exercising the same
code paths that a full experiment would use:

* Download (or reuse) the PAMAP2 archive on demand.
* Load subject-specific CSV files into a :class:`pandas.DataFrame`.
* Generate sliding windows, summarise them, and binarise the statistics.
* Train/test split by subject to respect the standard evaluation protocol.

Usage example
-------------
::

   python examples/pamap2_tsetlin.py \
       --data-root ./data/pamap2 \
       --train-subjects 101 103 105 \
       --test-subjects 109 \
       --epochs 10

The defaults intentionally keep the configuration small. For real experiments,
consider increasing the number of clauses, epochs, and the diversity of
subjects.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.pamap2 import (  # noqa: E402  - runtime path adjustment
    DEFAULT_SENSOR_COLUMNS,
    Pamap2WindowDataset,
    binarize_windows,
    ensure_pamap2_dataset,
    load_pamap2_dataframe,
    summarise_windows,
)
from pyTsetlinMachine.tm import MultiClassTsetlinMachine  # noqa: E402


@dataclass
class ExperimentConfig:
    """Configuration parameters captured alongside the results."""

    data_root: Path
    train_subjects: Sequence[int]
    test_subjects: Sequence[int]
    window_size: int
    step: int
    clauses: int
    threshold: int
    specificity: float
    epochs: int
    feature_columns: Sequence[str]


@dataclass
class ExperimentResult:
    """Lightweight container for serialising experiment outputs."""

    config: ExperimentConfig
    train_accuracy: float
    test_accuracy: float
    n_train_samples: int
    n_test_samples: int

    def to_json(self) -> str:
        payload = {
            "config": asdict(self.config),
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "n_train_samples": self.n_train_samples,
            "n_test_samples": self.n_test_samples,
        }
        return json.dumps(payload, indent=2)


def _accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    if not targets:
        return float("nan")
    correct = sum(int(p == t) for p, t in zip(predictions, targets))
    return correct / len(targets)


def _prepare_dataset(
    data_root: Path,
    subjects: Sequence[int],
    *,
    window_size: int,
    step: int,
    feature_columns: Sequence[str],
    thresholds=None,
) -> tuple[Pamap2WindowDataset, Sequence[int]]:
    frame = load_pamap2_dataframe(data_root, subjects=subjects)
    windows = summarise_windows(
        frame,
        window_size=window_size,
        step=step,
        feature_columns=feature_columns,
    )
    binary, thresholds_out = binarize_windows(windows.features, thresholds=thresholds)
    windows = Pamap2WindowDataset(features=binary, labels=windows.labels, metadata=windows.metadata)
    return windows, thresholds_out


def main(argv: Sequence[str] | None = None) -> ExperimentResult:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data/pamap2"),
        help="Directory used to cache/download the PAMAP2 archive",
    )
    parser.add_argument("--train-subjects", nargs="*", default=[101, 103], type=int)
    parser.add_argument("--test-subjects", nargs="*", default=[109], type=int)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--step", type=int, default=256)
    parser.add_argument("--clauses", type=int, default=60)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--specificity", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=list(DEFAULT_SENSOR_COLUMNS),
        help="Optional override for the PAMAP2 sensor columns to summarise",
    )

    args = parser.parse_args(argv)

    dataset_root = ensure_pamap2_dataset(args.data_root)

    config = ExperimentConfig(
        data_root=args.data_root,
        train_subjects=args.train_subjects,
        test_subjects=args.test_subjects,
        window_size=args.window_size,
        step=args.step,
        clauses=args.clauses,
        threshold=args.threshold,
        specificity=args.specificity,
        epochs=args.epochs,
        feature_columns=tuple(args.feature_columns),
    )

    train_windows, thresholds = _prepare_dataset(
        dataset_root,
        args.train_subjects,
        window_size=args.window_size,
        step=args.step,
        feature_columns=args.feature_columns,
    )

    test_windows, _ = _prepare_dataset(
        dataset_root,
        args.test_subjects,
        window_size=args.window_size,
        step=args.step,
        feature_columns=args.feature_columns,
        thresholds=thresholds,
    )

    model = MultiClassTsetlinMachine(args.clauses, args.threshold, args.specificity)
    model.fit(train_windows.features.tolist(), train_windows.labels.tolist(), epochs=args.epochs)

    train_predictions = model.predict(train_windows.features.tolist())
    test_predictions = model.predict(test_windows.features.tolist())

    train_accuracy = _accuracy(train_predictions, train_windows.labels.tolist())
    test_accuracy = _accuracy(test_predictions, test_windows.labels.tolist())

    result = ExperimentResult(
        config=config,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        n_train_samples=len(train_windows.labels),
        n_test_samples=len(test_windows.labels),
    )

    print(result.to_json())
    return result


if __name__ == "__main__":  # pragma: no cover - manual invocation entry-point
    main()
