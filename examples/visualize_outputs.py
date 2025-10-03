"""Generate grayscale JPEG visualizations for saved Tsetlin Machine time-series outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence

from simplecanvas import SimpleCanvas
from simplejpeg import save_jpeg


@dataclass
class GeneratedVisualization:
    filename: str
    description: str
    width: int
    height: int
    data: bytes


def read_accuracy_history(path: Path) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            history.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "test_accuracy": float(row["test_accuracy"]),
                }
            )
    return history


def read_predictions(path: Path) -> List[Dict[str, int]]:
    records: List[Dict[str, int]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(
                {
                    "true_label": int(row["true_label"]),
                    "predicted_label": int(row["predicted_label"]),
                }
            )
    return records


def read_sample_windows(path: Path) -> List[Dict[str, str]]:
    windows: List[Dict[str, str]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            windows.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "label": int(row["label"]),
                    "window_bits": row["window_bits"],
                }
            )
    return windows


def create_accuracy_plot(
    history: Sequence[Dict[str, float]],
    output_path: Path,
    keep_file: bool = True,
) -> GeneratedVisualization | None:
    if not history:
        return None

    history = sorted(history, key=lambda item: item["epoch"])
    width, height = 900, 540
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    canvas = SimpleCanvas(width, height)

    min_epoch = history[0]["epoch"]
    max_epoch = history[-1]["epoch"]
    epoch_span = max(1, max_epoch - min_epoch)

    def x_pos(epoch: float) -> int:
        return margin_left + int(round(((epoch - min_epoch) / epoch_span) * plot_width))

    def y_pos(value: float) -> int:
        clamped = max(0.0, min(1.0, value))
        return margin_top + int(round((1.0 - clamped) * plot_height))

    # Grid lines and labels for accuracy axis
    for i in range(11):
        value = i / 10.0
        y = y_pos(value)
        canvas.draw_horizontal_line(margin_left, margin_left + plot_width, y, 225)
        label = f"{value:.1f}"
        label_width = canvas.measure_text(label, scale=1)
        canvas.draw_text(margin_left - label_width - 12, y - 4, label, 0, scale=1)

    # Epoch ticks
    tick_count = min(len(history), 8)
    tick_step = max(1, epoch_span // max(1, tick_count - 1))
    epoch_ticks = list(range(min_epoch, max_epoch + 1, tick_step))
    if epoch_ticks[-1] != max_epoch:
        epoch_ticks.append(max_epoch)
    for epoch in epoch_ticks:
        x = x_pos(epoch)
        canvas.draw_vertical_line(x, margin_top, margin_top + plot_height, 240)
        label = str(epoch)
        label_width = canvas.measure_text(label, scale=1)
        canvas.draw_text(x - label_width // 2, margin_top + plot_height + 12, label, 0, scale=1)

    # Axes
    canvas.draw_line(margin_left, margin_top, margin_left, margin_top + plot_height, 0)
    canvas.draw_line(margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height, 0)

    def build_points(key: str) -> List[tuple[int, int]]:
        return [(x_pos(item["epoch"]), y_pos(item[key])) for item in history]

    train_points = build_points("train_accuracy")
    test_points = build_points("test_accuracy")

    canvas.draw_polyline(train_points, 40)
    canvas.draw_polyline(test_points, 120)

    for x, y in train_points:
        canvas.fill_rect(x - 2, y - 2, x + 2, y + 2, 40)
    for x, y in test_points:
        canvas.fill_rect(x - 2, y - 2, x + 2, y + 2, 140)

    canvas.draw_text(margin_left, margin_top - 40, "TRAIN VS TEST ACCURACY", 0, scale=2)
    canvas.draw_text(margin_left + plot_width // 2 - canvas.measure_text("EPOCH", scale=2) // 2,
                     margin_top + plot_height + 40,
                     "EPOCH",
                     0,
                     scale=2)
    canvas.draw_text(margin_left - 70, margin_top - 10, "ACCURACY", 0, scale=1)

    legend_x = width - margin_right - 200
    legend_y = margin_top + 10
    legend_w = 170
    legend_h = 70
    canvas.fill_rect(legend_x, legend_y, legend_x + legend_w, legend_y + legend_h, 245)
    canvas.draw_rect(legend_x, legend_y, legend_x + legend_w, legend_y + legend_h, 180)
    canvas.draw_line(legend_x + 15, legend_y + 20, legend_x + 60, legend_y + 20, 40)
    canvas.fill_rect(legend_x + 36, legend_y + 18, legend_x + 39, legend_y + 21, 40)
    canvas.draw_text(legend_x + 75, legend_y + 12, "TRAIN", 0, scale=1)
    canvas.draw_line(legend_x + 15, legend_y + 44, legend_x + 60, legend_y + 44, 120)
    canvas.fill_rect(legend_x + 36, legend_y + 42, legend_x + 39, legend_y + 45, 140)
    canvas.draw_text(legend_x + 75, legend_y + 36, "TEST", 0, scale=1)

    jpeg_bytes = save_jpeg(canvas.to_pixels(), output_path if keep_file else None)

    return GeneratedVisualization(
        filename=output_path.name,
        description="Train vs test accuracy across epochs.",
        width=canvas.width,
        height=canvas.height,
        data=jpeg_bytes,
    )


def create_confusion_matrix(
    predictions: Sequence[Dict[str, int]],
    output_path: Path,
    keep_file: bool = True,
) -> GeneratedVisualization | None:
    if not predictions:
        return None

    labels = sorted({row["true_label"] for row in predictions} | {row["predicted_label"] for row in predictions})
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)

    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for row in predictions:
        matrix[index[row["true_label"]]][index[row["predicted_label"]]] += 1

    cell_size = 60
    margin_left = 140
    margin_top = 80
    width = margin_left + cell_size * size + 80
    height = margin_top + cell_size * size + 120

    canvas = SimpleCanvas(width, height)
    canvas.draw_text(margin_left, margin_top - 50, "CONFUSION MATRIX", 0, scale=2)
    canvas.draw_text(margin_left + cell_size * size // 2 - canvas.measure_text("PREDICTED", scale=1) // 2,
                     margin_top + cell_size * size + 30,
                     "PREDICTED",
                     0,
                     scale=1)
    canvas.draw_text(margin_left - 100, margin_top - 20, "TRUE", 0, scale=1)

    max_value = max(max(row) for row in matrix) or 1
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            x0 = margin_left + col_idx * cell_size
            y0 = margin_top + row_idx * cell_size
            shade = 255 - int(190 * (value / max_value))
            canvas.fill_rect(x0, y0, x0 + cell_size - 2, y0 + cell_size - 2, shade)
            canvas.draw_rect(x0, y0, x0 + cell_size - 2, y0 + cell_size - 2, 100)
            label = str(value)
            text_width = canvas.measure_text(label, scale=1)
            canvas.draw_text(x0 + (cell_size - text_width) // 2, y0 + cell_size // 2 - 4, label, 0, scale=1)

    for idx, label in enumerate(labels):
        text = str(label)
        text_width = canvas.measure_text(text, scale=1)
        y = margin_top + idx * cell_size + cell_size // 2 - 4
        canvas.draw_text(margin_left - text_width - 20, y, text, 0, scale=1)
        x = margin_left + idx * cell_size + cell_size // 2 - text_width // 2
        canvas.draw_text(x, margin_top + cell_size * size + 60, text, 0, scale=1)

    jpeg_bytes = save_jpeg(canvas.to_pixels(), output_path if keep_file else None)

    return GeneratedVisualization(
        filename=output_path.name,
        description="Confusion matrix on held-out evaluation data.",
        width=canvas.width,
        height=canvas.height,
        data=jpeg_bytes,
    )


def create_sample_windows(
    samples: Sequence[Dict[str, str]],
    output_path: Path,
    max_windows: int,
    keep_file: bool = True,
) -> GeneratedVisualization | None:
    subset = list(samples[:max_windows])
    if not subset:
        return None

    bit_length = len(subset[0]["window_bits"])
    cell_size = 14
    margin_left = 140
    margin_top = 80
    width = margin_left + bit_length * cell_size + 80
    height = margin_top + len(subset) * cell_size + 120

    canvas = SimpleCanvas(width, height)
    canvas.draw_text(margin_left, margin_top - 50, "SAMPLE WINDOW BIT PATTERNS", 0, scale=2)

    for row_idx, record in enumerate(subset):
        bits = record["window_bits"]
        y0 = margin_top + row_idx * cell_size
        label_text = f"{record['sample_index']} ({record['label']})"
        canvas.draw_text(20, y0 + cell_size // 2 - 4, label_text, 0, scale=1)
        for bit_idx, bit in enumerate(bits):
            x0 = margin_left + bit_idx * cell_size
            fill = 60 if bit == "1" else 235
            canvas.fill_rect(x0, y0, x0 + cell_size - 3, y0 + cell_size - 3, fill)
            canvas.draw_rect(x0, y0, x0 + cell_size - 3, y0 + cell_size - 3, 160)

    for bit_idx in range(bit_length):
        if bit_idx % 5 == 0 or bit_idx == bit_length - 1:
            label = str(bit_idx)
            x = margin_left + bit_idx * cell_size
            canvas.draw_vertical_line(x, margin_top - 4, margin_top + len(subset) * cell_size, 200)
            text_width = canvas.measure_text(label, scale=1)
            canvas.draw_text(x - text_width // 2, margin_top + len(subset) * cell_size + 12, label, 0, scale=1)

    legend_x = margin_left
    legend_y = margin_top + len(subset) * cell_size + 50
    canvas.fill_rect(legend_x, legend_y, legend_x + 200, legend_y + 50, 245)
    canvas.draw_rect(legend_x, legend_y, legend_x + 200, legend_y + 50, 180)
    canvas.fill_rect(legend_x + 10, legend_y + 12, legend_x + 34, legend_y + 36, 60)
    canvas.draw_rect(legend_x + 10, legend_y + 12, legend_x + 34, legend_y + 36, 120)
    canvas.draw_text(legend_x + 46, legend_y + 18, "BIT = 1", 0, scale=1)
    canvas.fill_rect(legend_x + 110, legend_y + 12, legend_x + 134, legend_y + 36, 235)
    canvas.draw_rect(legend_x + 110, legend_y + 12, legend_x + 134, legend_y + 36, 120)
    canvas.draw_text(legend_x + 146, legend_y + 18, "BIT = 0", 0, scale=1)

    jpeg_bytes = save_jpeg(canvas.to_pixels(), output_path if keep_file else None)

    description = (
        "Bit-pattern heatmap for the first "
        f"{len(subset)} sample windows (max_windows={max_windows})."
    )

    return GeneratedVisualization(
        filename=output_path.name,
        description=description,
        width=canvas.width,
        height=canvas.height,
        data=jpeg_bytes,
    )


def visualize_run(
    run_dir: Path,
    output_dir: Path,
    max_windows: int,
    manifest_path: Path | None = None,
    keep_jpeg_files: bool = True,
) -> None:
    history = read_accuracy_history(run_dir / "accuracy_history.csv")
    predictions = read_predictions(run_dir / "test_predictions.csv")
    windows = read_sample_windows(run_dir / "sample_windows.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[GeneratedVisualization] = []

    result = create_accuracy_plot(
        history, output_dir / "accuracy_history.jpg", keep_file=keep_jpeg_files
    )
    if result:
        generated.append(result)

    result = create_confusion_matrix(
        predictions, output_dir / "confusion_matrix.jpg", keep_file=keep_jpeg_files
    )
    if result:
        generated.append(result)

    result = create_sample_windows(
        windows,
        output_dir / "sample_windows.jpg",
        max_windows,
        keep_file=keep_jpeg_files,
    )
    if result:
        generated.append(result)

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_images = [
            {
                "filename": item.filename,
                "description": item.description,
                "width": item.width,
                "height": item.height,
                "encoding": "base64",
                "data": base64.b64encode(item.data).decode("ascii"),
            }
            for item in generated
        ]

        manifest = {
            "run_directory": str(run_dir),
            "source_files": {
                "accuracy_history": "accuracy_history.csv",
                "test_predictions": "test_predictions.csv",
                "sample_windows": "sample_windows.csv",
            },
            "max_windows": max_windows,
            "images": encoded_images,
        }

        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if not keep_jpeg_files and generated:
        print(
            "Binary JPEG files were not written to disk because binary files are not supported "
            "for this repository. Use the JSON manifest to access the visualization bytes."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing exported time-series run data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store generated visualizations. Defaults to run_dir/visualizations.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=30,
        help="Maximum number of sample windows to visualize.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional path for a JSON artifact manifest containing base64-encoded JPEG data. "
            "Defaults to <output_dir>/visualizations_manifest.json."
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Disable writing the artifact manifest.",
    )
    parser.add_argument(
        "--skip-jpeg-files",
        action="store_true",
        help="Do not keep the generated JPEG files on disk (only emit the JSON manifest).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "visualizations"
    manifest_path: Path | None
    if args.no_manifest:
        manifest_path = None
    else:
        manifest_path = args.manifest or output_dir / "visualizations_manifest.json"

    visualize_run(
        run_dir,
        output_dir,
        max_windows=args.max_windows,
        manifest_path=manifest_path,
        keep_jpeg_files=not args.skip_jpeg_files,
    )


if __name__ == "__main__":
    main()
