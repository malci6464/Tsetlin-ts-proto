# High-frequency Sensor Benchmark: PAMAP2 Activity Monitoring

This note proposes the **PAMAP2 Physical Activity Monitoring** dataset as a
foundation for comparing a Tsetlin Machine pipeline with transformer-style
temporal models on dense wearable sensor streams. It consolidates the
practical details needed to pull the data into this repository and summarises
published transformer/LSTM results that can anchor a baseline section for a
paper draft.

## Dataset snapshot

- **Domain:** multi-sensor human activity recognition collected from three
  inertial measurement units (hand, chest, ankle) plus a heart-rate monitor.
- **Sampling rate:** 100 Hz for the accelerometer/gyroscope/magnetometer
  channels; the heart-rate stream is downsampled to roughly 9 Hz.
- **Subjects & activities:** 9 participants performing 18 labeled activities
  (daily living and sports), yielding ~12 hours of recordings.
- **File layout:** each subject has a separate CSV file with synchronized sensor
  channels and activity identifiers. The official splits separate train/test by
  subject, which aligns with leave-one-subject-out evaluation used in the
  literature.
- **Availability:** downloadable from the
  [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/242/pamap2+physical+activity+monitoring)
  or via mirrored Kaggle projects (e.g. `pamap2-physical-activity-monitoring`).
  The original release is licensed for academic use, matching this prototype's
  research focus.

### Integration considerations

1. **Storage footprint:** the full dataset is ~1.1 GB uncompressed. A minimal
   subset (e.g. one IMU triplet plus heart rate) can be extracted to keep the
   repository lightweight, while documentation can point to a reproducible
   download script for full-scale experiments.
2. **Preprocessing:** published baselines standardize per-channel, apply a
   sliding window of 5.12 seconds (512 timesteps at 100 Hz) with 50% overlap,
   and discard idle segments. Implementing the windowing pipeline alongside the
   existing `examples/time_series_tsetlin.py` structure should allow quick
   experimentation.
3. **Label imbalance:** activities such as "rope jumping" are short, so
   stratified batching or class weighting will help the Tsetlin configuration.

## Established baselines to cite

- **Transformer:** *HARFormer: Transformers-based Human Activity Recognition*
  (Sensors, 2022) benchmarks a lightweight transformer encoder on PAMAP2,
  reporting macro-F1 gains over convolutional and bi-directional LSTM baselines
  under leave-one-subject-out evaluation. The architecture uses multi-head
  self-attention on flattened sensor windows with positional encodings.
- **LSTM:** The *DeepConvLSTM* architecture (Ordóñez & Roggen, 2016) remains a
  strong recurrent baseline on PAMAP2, mixing convolutional feature extractors
  with stacked LSTM layers. Later studies (e.g. the above HARFormer paper)
  reproduce DeepConvLSTM as a comparison point, providing reference accuracy
  values around the low-to-mid 90% macro-F1 range.
- **Time-series transformer variants:** Follow-up work such as *Self-Supervised
  Transformers for Wearable Activity Recognition* (Khedher et al., 2023)
  explores masked modeling objectives on the same dataset, offering additional
  pretraining baselines that we can mention when motivating future extensions.

Collecting these references ensures that a "Related Work" or "Baseline"
section in a paper draft can cite both transformer and LSTM results directly
comparable to any Tsetlin Machine experiments executed within this repository.

## Next steps for this repository

1. **Data ingestion script:** add a utility (e.g. `scripts/download_pamap2.py`)
   that fetches and unpacks the dataset, optionally trimming to a curated subset
   for CI-friendly experiments. *(Update: `datasets/pamap2.py` now implements
   runtime download and windowing helpers together with an executable example
   in `examples/pamap2_tsetlin.py`.)*
2. **Notebook or pipeline prototype:** adapt `examples/time_series_tsetlin.py`
   to load the PAMAP2 windows and train a multiclass Tsetlin Machine, mirroring
   the evaluation protocol from the transformer papers.
3. **Result tracking:** extend `docs/time_series_outputs` with a PAMAP2 page
   similar to the existing maritime and temperature scenarios. This will give us
   a place to visualise accuracy curves and sample confusion matrices once the
   experiments run.
4. **Comparative analysis:** outline a reproducible transformer baseline
   (PyTorch implementation of HARFormer or DeepConvLSTM) so the repo can produce
   side-by-side metrics for the eventual paper.

Keeping these tasks scoped and documented positions the project to articulate
"how far off" the Tsetlin Machine is from state-of-the-art transformer
approaches on high-frequency wearable sensor data.
