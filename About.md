# Tsetlin Machine Prototype

This repository collects Python examples that demonstrate how to apply
Tsetlin Machines to different problem domains.

## Available examples

- `examples/time_series_tsetlin.py` now offers two synthetic scenarios:
  the original binary motif detection task and a maritime sensor anomaly
  detector. In both cases the script trains a `MultiClassTsetlinMachine`,
  captures train/test accuracy per epoch, and persists CSV/JSON summaries.
  When Matplotlib is available the learning curves are rendered as PNG files;
  otherwise a textual fallback is emitted so the workflow remains runnable
  without network access.

## Research planning notes

- [High-frequency Sensor Benchmark: PAMAP2 Activity Monitoring](docs/research/high_frequency_sensor_plan.md)
  summarises a high-density wearable dataset with published transformer and
  LSTM baselines. The note now links to `datasets/pamap2.py` for runtime
  ingestion/windowing helpers and `examples/pamap2_tsetlin.py` for a CLI
  experiment that trains the surrogate Tsetlin Machine on PAMAP2 windows.
