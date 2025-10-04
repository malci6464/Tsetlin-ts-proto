"""Time-series pattern detection with a Tsetlin Machine.

This example provides two complementary synthetic datasets that you can use
to experiment with a Tsetlin Machine on sequential data:

``binary-pattern``
    Reproduces the classic sliding-window toy problem where the task is to
    recognise a particular binary motif under noisy conditions.

``maritime``
    Mimics aggregated maritime sensor alarms (wave height, hull vibration,
    etc.) and tasks the model with identifying hazardous vessel behaviour.

``industrial``
    Synthesises high-dimensional industrial telemetry (thousands of sensors)
    and evaluates the model's ability to raise alarms when multiple
    sub-systems breach safety thresholds simultaneously.

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
import time
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import tracemalloc
from statistics import mean

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


def generate_industrial_sensor_dataset(
    n_samples: int,
    n_sensors: int = 3000,
    hazard_probability: float = 0.18,
    noise_probability: float = 0.015,
    rng: random.Random | None = None,
) -> DatasetBundle:
    """Create a multi-sensor industrial anomaly detection dataset.

    Each sample represents a 10 second snapshot of more than three thousand
    binary threshold indicators gathered from different subsystems (engine,
    pumps, turbines, etc.).  Alarms are triggered by specific combinations of
    sensor breaches to emulate rule-based safety policies.
    """

    if rng is None:
        rng = random.Random()

    if n_sensors < 12:
        raise ValueError("n_sensors must be large enough to host key sensors (>=12)")

    sensor_names: List[str] = [
        "engine_temp_gt_950c",
        "engine_vibration_gt_60mm_s",
        "fuel_pressure_below_25psi",
        "coolant_pressure_below_30psi",
        "pump_vibration_gt_40mm_s",
        "exhaust_temp_gt_850c",
        "turbine_vibration_gt_55mm_s",
        "oil_pressure_below_20psi",
        "generator_temp_gt_120c",
        "bearing_vibration_gt_45mm_s",
        "hydraulic_temp_gt_95c",
        "hydraulic_pressure_below_2200psi",
    ]
    # Fill remaining sensors with generic telemetry channels.
    sensor_names.extend([f"auxiliary_sensor_{index:04d}" for index in range(len(sensor_names), n_sensors)])

    key_sensor_indices = {
        name: sensor_names.index(name)
        for name in [
            "engine_temp_gt_950c",
            "engine_vibration_gt_60mm_s",
            "coolant_pressure_below_30psi",
            "pump_vibration_gt_40mm_s",
            "exhaust_temp_gt_850c",
            "turbine_vibration_gt_55mm_s",
            "oil_pressure_below_20psi",
            "fuel_pressure_below_25psi",
            "generator_temp_gt_120c",
            "bearing_vibration_gt_45mm_s",
            "hydraulic_temp_gt_95c",
            "hydraulic_pressure_below_2200psi",
        ]
    }

    hazard_rules = [
        {
            "name": "Engine thermal runaway",
            "sensors": [
                key_sensor_indices["engine_temp_gt_950c"],
                key_sensor_indices["engine_vibration_gt_60mm_s"],
            ],
        },
        {
            "name": "Pump cavitation",
            "sensors": [
                key_sensor_indices["coolant_pressure_below_30psi"],
                key_sensor_indices["pump_vibration_gt_40mm_s"],
            ],
        },
        {
            "name": "Fuel starvation",
            "sensors": [
                key_sensor_indices["fuel_pressure_below_25psi"],
                key_sensor_indices["engine_temp_gt_950c"],
                key_sensor_indices["oil_pressure_below_20psi"],
            ],
        },
        {
            "name": "Turbine imbalance",
            "sensors": [
                key_sensor_indices["exhaust_temp_gt_850c"],
                key_sensor_indices["turbine_vibration_gt_55mm_s"],
                key_sensor_indices["generator_temp_gt_120c"],
            ],
        },
        {
            "name": "Hydraulic overheating",
            "sensors": [
                key_sensor_indices["hydraulic_temp_gt_95c"],
                key_sensor_indices["hydraulic_pressure_below_2200psi"],
                key_sensor_indices["bearing_vibration_gt_45mm_s"],
            ],
        },
    ]

    inputs: List[List[int]] = []
    targets: List[int] = []
    hazard_counts = {rule["name"]: 0 for rule in hazard_rules}

    for _ in range(n_samples):
        snapshot = [0] * n_sensors
        label = 0

        if rng.random() < hazard_probability:
            rule = rng.choice(hazard_rules)
            for sensor_index in rule["sensors"]:
                snapshot[sensor_index] = 1
            label = 1
            hazard_counts[rule["name"]] += 1
            # Additional correlated spill-over events.
            if rule["sensors"]:
                correlated = rng.sample(rule["sensors"], k=1)
                for sensor_index in correlated:
                    if sensor_index + 1 < n_sensors:
                        snapshot[sensor_index + 1] = 1
        else:
            # Inject soft warnings that resemble near-misses.
            if rng.random() < hazard_probability * 0.3:
                near_miss = rng.choice(hazard_rules)
                sampled = rng.sample(near_miss["sensors"], k=max(1, len(near_miss["sensors"]) - 1))
                for sensor_index in sampled:
                    snapshot[sensor_index] = 1

        # Background noise across the long tail of telemetry.
        for sensor_index in range(len(key_sensor_indices), n_sensors):
            if rng.random() < noise_probability:
                snapshot[sensor_index] = 1

        inputs.append(snapshot)
        targets.append(label)

    metadata: Dict[str, object] = {
        "type": "industrial",
        "sensor_count": int(n_sensors),
        "hazard_probability": float(hazard_probability),
        "noise_probability": float(noise_probability),
        "samples": int(n_samples),
        "hazard_rules": [rule["name"] for rule in hazard_rules],
        "rule_counts": hazard_counts,
        "snapshot_interval_seconds": 10,
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


def benchmark_inference(
    tm: MultiClassTsetlinMachine,
    inputs: Sequence[Sequence[int]],
    sample_interval_seconds: float,
    repetitions: int = 8,
) -> Dict[str, float]:
    """Benchmark inference throughput and memory usage for the given model."""

    tracemalloc.start()
    start_time = time.perf_counter()
    total_predictions = 0
    for _ in range(repetitions):
        tm.predict(inputs)
        total_predictions += len(inputs)
    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    per_sample = elapsed / total_predictions if total_predictions else float("nan")
    samples_per_second = total_predictions / elapsed if elapsed > 0 else float("inf")
    available_window = sample_interval_seconds
    utilisation = per_sample / available_window if available_window > 0 else float("nan")

    return {
        "repetitions": float(repetitions),
        "total_predictions": float(total_predictions),
        "elapsed_seconds": float(elapsed),
        "per_sample_seconds": float(per_sample),
        "samples_per_second": float(samples_per_second),
        "prediction_window_seconds": float(sample_interval_seconds),
        "window_utilisation": float(utilisation),
        "current_memory_bytes": float(current),
        "peak_memory_bytes": float(peak),
    }


def run_sensor_scaling_benchmark(
    sensor_counts: Sequence[int],
    base_rng: random.Random,
    n_clauses: int,
    threshold: int,
    s: float,
    base_epochs: int,
    sample_interval_seconds: float,
    n_samples: int = 1600,
    hazard_probability: float = 0.2,
    noise_probability: float = 0.02,
) -> List[Dict[str, float]]:
    """Train smaller industrial models and benchmark inference as sensors scale."""

    results: List[Dict[str, float]] = []
    scaling_epochs = max(1, min(base_epochs, 5))

    for sensor_count in sensor_counts:
        # Derive a deterministic seed for each configuration while remaining reproducible.
        seed = base_rng.randint(0, 1_000_000)
        local_rng = random.Random(seed)

        bundle = generate_industrial_sensor_dataset(
            n_samples=n_samples,
            n_sensors=sensor_count,
            hazard_probability=hazard_probability,
            noise_probability=noise_probability,
            rng=local_rng,
        )

        dataset = bundle.dataset
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.inputs, dataset.targets, 0.25, local_rng
        )

        tm = MultiClassTsetlinMachine(n_clauses=n_clauses, T=threshold, s=s)

        train_start = time.perf_counter()
        for _ in range(scaling_epochs):
            tm.fit(X_train, y_train, epochs=1, incremental=True)
        train_elapsed = time.perf_counter() - train_start

        y_pred = tm.predict(X_test)
        test_accuracy = _accuracy(y_pred, y_test)

        metrics = benchmark_inference(
            tm,
            dataset.inputs,
            sample_interval_seconds=sample_interval_seconds,
            repetitions=6,
        )

        result: Dict[str, float] = {
            "sensor_count": float(sensor_count),
            "samples": float(len(dataset.inputs)),
            "train_epochs": float(scaling_epochs),
            "train_time_seconds": float(train_elapsed),
            "test_accuracy": float(test_accuracy),
        }
        result.update(metrics)
        results.append(result)

    return results


def save_sensor_scaling_results(
    results: Sequence[Dict[str, float]], directory: Path
) -> Tuple[Path, Path]:
    """Persist the sensor scaling study to CSV and JSON files."""

    if not results:
        raise ValueError("Scaling results must contain at least one entry")

    csv_path = directory / "sensor_scaling.csv"
    json_path = directory / "sensor_scaling.json"

    fieldnames = [
        "sensor_count",
        "samples",
        "train_epochs",
        "train_time_seconds",
        "test_accuracy",
        "repetitions",
        "total_predictions",
        "elapsed_seconds",
        "per_sample_seconds",
        "samples_per_second",
        "prediction_window_seconds",
        "window_utilisation",
        "current_memory_bytes",
        "peak_memory_bytes",
    ]

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with json_path.open("w") as fh:
        json.dump(list(results), fh, indent=2)

    return csv_path, json_path


def plot_sensor_scaling(
    results: Sequence[Dict[str, float]], directory: Path
) -> Path | None:
    """Render a static plot for the sensor scaling study when Matplotlib is available."""

    if plt is None or not results:
        return None

    path = directory / "sensor_scaling.png"
    sensor_counts = [row.get("sensor_count", 0.0) for row in results]
    throughput = [row.get("samples_per_second", float("nan")) for row in results]
    latency_ms = [row.get("per_sample_seconds", float("nan")) * 1000 for row in results]
    memory_mib = [row.get("peak_memory_bytes", 0.0) / 1_048_576 for row in results]

    fig, ax1 = plt.subplots(figsize=(6.5, 3.5))  # type: ignore[call-arg]
    color_throughput = "tab:blue"
    color_latency = "tab:green"
    color_memory = "tab:red"

    ax1.set_xlabel("Active sensor channels")
    ax1.set_ylabel("Snapshots per second", color=color_throughput)
    ax1.plot(sensor_counts, throughput, marker="o", color=color_throughput, label="Throughput")
    ax1.tick_params(axis="y", labelcolor=color_throughput)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Latency (ms)", color=color_latency)
    ax2.plot(sensor_counts, latency_ms, marker="s", color=color_latency, label="Latency (ms)")
    ax2.tick_params(axis="y", labelcolor=color_latency)

    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))
    ax3.set_ylabel("Peak memory (MiB)", color=color_memory)
    ax3.plot(sensor_counts, memory_mib, marker="^", color=color_memory, label="Peak memory")
    ax3.tick_params(axis="y", labelcolor=color_memory)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # type: ignore[arg-type]

    return path


def save_inference_report(
    metrics: Dict[str, float],
    metadata: Dict[str, object],
    hyperparameters: Dict[str, object],
    directory: Path,
) -> Path:
    """Persist a Markdown report summarising inference scalability."""

    path = directory / "inference_report.md"
    sensor_count = metadata.get("sensor_count", "?")
    snapshot_interval = metadata.get("snapshot_interval_seconds", "?")
    clauses = hyperparameters.get("n_clauses", "?")

    samples_per_second = metrics.get("samples_per_second", float("nan"))
    per_sample_seconds = metrics.get("per_sample_seconds", float("nan"))
    window_utilisation = metrics.get("window_utilisation", float("nan"))
    peak_memory = metrics.get("peak_memory_bytes", float("nan"))

    lines = [
        "# Industrial Inference Scalability Report",
        "",
        f"* **Sensors evaluated:** {sensor_count}",
        f"* **Snapshot interval:** every {snapshot_interval} seconds",
        f"* **Tsetlin clauses (total rules):** {clauses}",
        f"* **Benchmark repetitions:** {int(metrics.get('repetitions', 0))}",
        f"* **Total predictions:** {int(metrics.get('total_predictions', 0))}",
        "",
        "## Throughput",
        f"* Average latency per snapshot: {per_sample_seconds:.6f} seconds",
        f"* Effective throughput: {samples_per_second:.2f} snapshots/second",
        f"* Fraction of 10 second budget used: {window_utilisation * 100:.2f}%",
        "",
        "## Memory usage",
        f"* Peak RSS during inference loop (measured via tracemalloc): {peak_memory / 1_048_576:.3f} MiB",
        f"* Current memory after benchmark: {metrics.get('current_memory_bytes', 0) / 1_048_576:.3f} MiB",
        "",
        "## Notes",
        "- Measurements include end-to-end evaluation of the trained Tsetlin Machine",
        "  across thousands of binary sensor indicators using repeated inference passes.",
        "- The utilisation figure compares average latency with the 10 second arrival",
        "  interval, showing ample headroom for real-time operation.",
    ]

    with path.open("w") as fh:
        fh.write("\n".join(lines))

    return path


def run_demo(
    epochs: int,
    n_clauses: int,
    threshold: int,
    s: float,
    seed: int,
    scenario: str,
    output_dir: str | Path | None,
    run_scaling_study: bool,
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
    elif scenario == "industrial":
        bundle = generate_industrial_sensor_dataset(
            n_samples=2400,
            n_sensors=3000,
            hazard_probability=0.2,
            noise_probability=0.02,
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

    if scenario == "industrial":
        metrics = benchmark_inference(
            tm,
            dataset.inputs,
            sample_interval_seconds=float(metadata.get("snapshot_interval_seconds", 10)),
            repetitions=10,
        )
        report_path = save_inference_report(metrics, metadata, hyperparameters, output_path)
        print("Inference benchmark completed:", report_path.resolve())
        print(
            "Average latency per snapshot:",
            f"{metrics['per_sample_seconds']:.6f}s",
            "| throughput:",
            f"{metrics['samples_per_second']:.2f} samples/s",
        )

        if run_scaling_study:
            print("Running sensor scaling study across feature counts...")
            sensor_counts = sorted({
                int(metadata.get("sensor_count", 3000) // 4),
                int(metadata.get("sensor_count", 3000) // 2),
                int(metadata.get("sensor_count", 3000) * 3 // 4),
                int(metadata.get("sensor_count", 3000)),
            })
            sensor_counts = [count for count in sensor_counts if count >= 12]
            scaling_results = run_sensor_scaling_benchmark(
                sensor_counts,
                base_rng=random.Random(seed + 42),
                n_clauses=n_clauses,
                threshold=threshold,
                s=s,
                base_epochs=epochs,
                sample_interval_seconds=float(metadata.get("snapshot_interval_seconds", 10)),
            )
            csv_path, json_path = save_sensor_scaling_results(scaling_results, output_path)
            plot_sensor_scaling(scaling_results, output_path)
            mean_latency_ms = mean(result["per_sample_seconds"] * 1000 for result in scaling_results)
            print(
                "Scaling study saved to:",
                csv_path.name,
                "and",
                json_path.name,
                f"| average latency: {mean_latency_ms:.3f} ms",
            )

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
        choices=("binary-pattern", "maritime", "industrial"),
        default="binary-pattern",
        help="Synthetic dataset to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory where artefacts should be stored",
    )
    parser.add_argument(
        "--skip-scaling-study",
        action="store_true",
        help="Disable the industrial sensor scaling benchmark",
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
        run_scaling_study=not args.skip_scaling_study,
    )


if __name__ == "__main__":
    main()
