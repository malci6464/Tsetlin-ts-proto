"""Time-series pattern detection with a Tsetlin Machine.

This example provides two complementary synthetic datasets that you can use
to experiment with a Tsetlin Machine on sequential data:

``binary-pattern``
    Reproduces the classic sliding-window toy problem where the task is to
    recognise a particular binary motif under noisy conditions.

``maritime``
    Mimics aggregated maritime sensor alarms (wave height, hull vibration,
    etc.) and tasks the model with identifying hazardous vessel behaviour.

For each run the script records the training and test accuracy after every
epoch, saves the results to CSV/JSON files, and attempts to render a plot of
the learning dynamics. When Matplotlib is unavailable (e.g. in offline
evaluation environments) the plot step gracefully degrades to a textual
summary so that the workflow can still be executed end-to-end.

Usage
-----
The example ships with a lightweight, perceptron-based fallback implementation
of :class:`pyTsetlinMachine.tm.MultiClassTsetlinMachine`. This allows the
script to run without external dependencies, but you can install the official
package if you wish:

>>> pip install pyTsetlinMachine matplotlib

Then run this script (the defaults reproduce the ``binary-pattern`` example):

>>> python examples/time_series_tsetlin.py --scenario maritime --epochs 40
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyTsetlinMachine.tm import MultiClassTsetlinMachine

_MATPLOTLIB_SPEC = importlib.util.find_spec("matplotlib")
if _MATPLOTLIB_SPEC is not None:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt  # type: ignore
else:  # pragma: no cover - exercised in minimal environments
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]


@dataclass
class Dataset:
    """Container for the windowed time-series dataset."""

    inputs: List[List[int]]
    targets: List[int]


@dataclass
class DatasetBundle:
    """Dataset together with metadata useful for analysis and persistence."""

    dataset: Dataset
    metadata: Dict[str, object]


@dataclass
class TrainingHistory:
    """Accuracy statistics gathered during training."""

    epochs: List[int]
    train_accuracy: List[float]
    test_accuracy: List[float]


def _xor_bit(bit: int, flip: bool) -> int:
    return bit ^ int(flip)


def _flatten(window_steps: Sequence[Sequence[int]]) -> List[int]:
    return [value for step in window_steps for value in step]


def _accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    total = len(targets)
    if total == 0:
        return float("nan")
    correct = sum(1 for pred, truth in zip(predictions, targets) if int(pred) == int(truth))
    return correct / total


def generate_binary_pattern_dataset(
    n_sequences: int,
    sequence_length: int,
    pattern: Sequence[int],
    noise_probability: float = 0.05,
    pattern_probability: float = 0.6,
    rng: random.Random | None = None,
) -> DatasetBundle:
    """Create the classic sliding-window binary pattern dataset."""

    if rng is None:
        rng = random.Random()

    window_size = len(pattern)
    windows: List[List[int]] = []
    labels: List[int] = []

    positives = 0

    for _ in range(n_sequences):
        sequence = [1 if rng.random() < 0.5 else 0 for _ in range(sequence_length)]

        if rng.random() < pattern_probability:
            start = rng.randrange(0, sequence_length - window_size + 1)
            for offset, value in enumerate(pattern):
                sequence[start + offset] = int(value)

        noisy_sequence = [_xor_bit(bit, rng.random() < noise_probability) for bit in sequence]

        for index in range(sequence_length - window_size + 1):
            window = noisy_sequence[index : index + window_size]
            label = int(all(window[pos] == int(pattern[pos]) for pos in range(window_size)))
            windows.append(window.copy())
            labels.append(label)
            positives += label

    inputs = windows
    targets = labels

    metadata: Dict[str, object] = {
        "type": "binary-pattern",
        "pattern": [int(value) for value in pattern],
        "window_size": int(window_size),
        "noise_probability": float(noise_probability),
        "pattern_probability": float(pattern_probability),
        "n_sequences": int(n_sequences),
        "sequence_length": int(sequence_length),
        "positives": int(positives),
        "samples": int(len(inputs)),
    }
    return DatasetBundle(Dataset(inputs, targets), metadata)


def generate_maritime_sensor_dataset(
    n_sequences: int,
    sequence_length: int,
    window_size: int = 5,
    noise_probability: float = 0.08,
    event_probability: float = 0.45,
    rng: random.Random | None = None,
) -> DatasetBundle:
    """Simulate maritime sensor alerts and extract windowed binary features."""

    if rng is None:
        rng = random.Random()

    sensors = [
        "wave_height_gt_6m",
        "surface_temp_below_12c",
        "hull_vibration_gt_70",
        "barometric_drop_gt_8hpa",
    ]

    hazard_pattern: List[List[int]] = [
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 1, 1],
    ]

    if window_size != len(hazard_pattern):
        raise ValueError("window_size must match hazard pattern length (5)")

    windows: List[List[int]] = []
    labels: List[int] = []
    positives = 0

    for _ in range(n_sequences):
        steps: List[List[int]] = []
        event_flags = [False] * sequence_length
        event_start = -1

        if rng.random() < event_probability:
            event_start = rng.randrange(0, sequence_length - window_size + 1)
            for idx in range(event_start, event_start + window_size):
                event_flags[idx] = True

        for index in range(sequence_length):
            if event_flags[index]:
                pattern_index = index - event_start
                features = hazard_pattern[pattern_index]
            else:
                features = [
                    int(rng.random() < 0.18),  # occasional large waves
                    int(rng.random() < 0.12),  # colder currents
                    int(rng.random() < 0.15),  # vibration spikes
                    int(rng.random() < 0.1),  # pressure swings
                ]

            noisy_step = [_xor_bit(bit, rng.random() < noise_probability) for bit in features]
            steps.append(noisy_step)

        for index in range(sequence_length - window_size + 1):
            window_steps = steps[index : index + window_size]
            window = _flatten(window_steps)
            label = int(all(event_flags[idx] for idx in range(index, index + window_size)))
            windows.append(window)
            labels.append(label)
            positives += label

    inputs = windows
    targets = labels

    metadata: Dict[str, object] = {
        "type": "maritime",
        "sensors": sensors,
        "window_size": int(window_size),
        "noise_probability": float(noise_probability),
        "event_probability": float(event_probability),
        "n_sequences": int(n_sequences),
        "sequence_length": int(sequence_length),
        "hazard_pattern": [value for row in hazard_pattern for value in row],
        "positives": int(positives),
        "samples": int(len(inputs)),
    }
    return DatasetBundle(Dataset(inputs, targets), metadata)


def train_test_split(
    inputs: Sequence[Sequence[int]],
    targets: Sequence[int],
    test_size: float,
    rng: random.Random,
) -> Tuple[List[List[int]], List[List[int]], List[int], List[int]]:
    """Split inputs and targets into train and test subsets."""

    n_samples = len(inputs)
    indices = list(range(n_samples))
    rng.shuffle(indices)
    split_index = int(n_samples * (1 - test_size))

    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    X_train = [list(inputs[idx]) for idx in train_idx]
    X_test = [list(inputs[idx]) for idx in test_idx]
    y_train = [int(targets[idx]) for idx in train_idx]
    y_test = [int(targets[idx]) for idx in test_idx]
    return X_train, X_test, y_train, y_test


def prepare_output_directory(scenario: str, output_dir: str | Path | None) -> Path:
    """Create and return the directory for persisting run artefacts."""

    if output_dir is None:
        base = Path("time_series_outputs") / scenario
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base / timestamp
    else:
        path = Path(output_dir)

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_history(history: TrainingHistory, directory: Path) -> Path:
    path = directory / "accuracy_history.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_accuracy", "test_accuracy"])
        for epoch, train_acc, test_acc in zip(
            history.epochs, history.train_accuracy, history.test_accuracy
        ):
            writer.writerow([epoch, f"{train_acc:.6f}", f"{test_acc:.6f}"])
    return path


def save_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    directory: Path,
    filename: str = "test_predictions.csv",
) -> Path:
    path = directory / filename
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_index", "true_label", "predicted_label"])
        for index, (truth, prediction) in enumerate(zip(y_true, y_pred)):
            writer.writerow([index, int(truth), int(prediction)])
    return path


def save_sample_windows(
    inputs: Sequence[Sequence[int]],
    targets: Sequence[int],
    directory: Path,
    n_samples: int = 12,
    filename: str = "sample_windows.csv",
) -> Path:
    path = directory / filename
    subset = min(n_samples, len(inputs))
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_index", "label", "window_bits"])
        for index in range(subset):
            window_bits = "".join(str(int(bit)) for bit in inputs[index])
            writer.writerow([index, int(targets[index]), window_bits])
    return path


def plot_history(history: TrainingHistory, directory: Path) -> Path:
    if plt is None:
        path = directory / "accuracy_curve.txt"
        with path.open("w") as fh:
            fh.write("Epoch,Train accuracy,Test accuracy\n")
            for epoch, train_acc, test_acc in zip(
                history.epochs, history.train_accuracy, history.test_accuracy
            ):
                fh.write(f"{epoch},{train_acc:.6f},{test_acc:.6f}\n")
        return path

    path = directory / "accuracy_curve.png"
    fig, ax = plt.subplots(figsize=(6.0, 3.5))  # type: ignore[call-arg]
    ax.plot(history.epochs, history.train_accuracy, marker="o", label="Train accuracy")
    ax.plot(history.epochs, history.test_accuracy, marker="s", label="Test accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # type: ignore[arg-type]
    return path


def save_run_summary(
    directory: Path,
    scenario: str,
    metadata: Dict[str, object],
    hyperparameters: Dict[str, object],
    history: TrainingHistory,
    final_test_accuracy: float,
) -> Path:
    summary = {
        "scenario": scenario,
        "metadata": metadata,
        "hyperparameters": hyperparameters,
        "training": {
            "epochs": history.epochs,
            "train_accuracy": history.train_accuracy,
            "test_accuracy": history.test_accuracy,
            "final_test_accuracy": final_test_accuracy,
            "best_test_accuracy": max(history.test_accuracy) if history.test_accuracy else None,
        },
    }

    path = directory / "run_summary.json"
    with path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    return path


def run_demo(
    epochs: int,
    n_clauses: int,
    threshold: int,
    s: float,
    seed: int,
    scenario: str,
    output_dir: str | Path | None,
) -> Path:
    """Train and evaluate the Tsetlin Machine on the requested dataset."""

    rng = random.Random(seed)

    if scenario == "binary-pattern":
        pattern = [1, 1, 0, 1, 0]
        bundle = generate_binary_pattern_dataset(
            n_sequences=200,
            sequence_length=25,
            pattern=pattern,
            noise_probability=0.1,
            pattern_probability=0.7,
            rng=rng,
        )
    elif scenario == "maritime":
        bundle = generate_maritime_sensor_dataset(
            n_sequences=180,
            sequence_length=30,
            window_size=5,
            noise_probability=0.09,
            event_probability=0.5,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown scenario '{scenario}'")

    dataset = bundle.dataset
    metadata = bundle.metadata

    X_train, X_test, y_train, y_test = train_test_split(dataset.inputs, dataset.targets, 0.25, rng)

    tm = MultiClassTsetlinMachine(n_clauses=n_clauses, T=threshold, s=s)

    history = TrainingHistory([], [], [])

    for epoch in range(1, epochs + 1):
        tm.fit(X_train, y_train, epochs=1, incremental=True)
        train_predictions = tm.predict(X_train)
        test_predictions = tm.predict(X_test)
        train_accuracy = _accuracy(train_predictions, y_train)
        test_accuracy = _accuracy(test_predictions, y_test)
        history.epochs.append(epoch)
        history.train_accuracy.append(train_accuracy)
        history.test_accuracy.append(test_accuracy)
        print(
            f"Epoch {epoch:02d}: train accuracy={train_accuracy:.3f}, test accuracy={test_accuracy:.3f}"
        )

    y_pred = tm.predict(X_test)

    output_path = prepare_output_directory(scenario, output_dir)

    save_history(history, output_path)
    save_predictions(y_test, y_pred, output_path)
    save_sample_windows(dataset.inputs, dataset.targets, output_path)
    plot_history(history, output_path)

    hyperparameters = {
        "epochs": epochs,
        "n_clauses": n_clauses,
        "threshold": threshold,
        "s": s,
        "seed": seed,
        "test_size": 0.25,
    }
    final_test_accuracy = history.test_accuracy[-1] if history.test_accuracy else float("nan")
    save_run_summary(output_path, scenario, metadata, hyperparameters, history, final_test_accuracy)

    print(f"Final test accuracy: {final_test_accuracy:.3f}")
    print(f"Artefacts saved to: {output_path.resolve()}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Command line interface for the example script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument(
        "--clauses",
        type=int,
        default=60,
        help="Number of clauses per class in the Tsetlin Machine",
    )
    parser.add_argument(
        "--threshold", type=int, default=15, help="Voting threshold (T) for the Tsetlin Machine"
    )
    parser.add_argument(
        "--s", type=float, default=3.5, help="Specificity parameter s for the Tsetlin Machine"
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducible results")
    parser.add_argument(
        "--scenario",
        choices=("binary-pattern", "maritime"),
        default="binary-pattern",
        help="Synthetic dataset to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory where artefacts should be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(
        epochs=args.epochs,
        n_clauses=args.clauses,
        threshold=args.threshold,
        s=args.s,
        seed=args.seed,
        scenario=args.scenario,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
