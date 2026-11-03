# Planned improvements

The current release retains the text/trajectory inference interface and the existing Train and Evaluation workflows. The following extensions are deferred to a later version.

- [ ] **Audio inference:** verify the official OmniHuMo waveform extractor, its pretrained checkpoint and feature conventions; provide waveform/feature input commands and speech/music examples. Validate audio-only and combined conditions with the released 3B checkpoint.
- [x] **Data preparation documentation:** document downloads, directory layout, record schema, normalization and split configuration in [Data preparation](docs/DATA_PREPARATION.md).
- [ ] **Data preparation tools:** add automated raw-data conversion, processed-record validation and split-list generation while preserving the official train/test separation.
- [ ] **Standalone checkpoints:** publish Stage2 3B, Stage3 3B, motion-tokenizer and TEMOS-evaluator files separately, with matching configurations, SHA256 checksums and pinned download commands. Verify downloads from a clean directory.
- [x] **Examples and visualization:** include an analytic trajectory, generated skeleton/MP4 preview and CPU skeleton renderer. See [Examples and visualization](examples/README.md).
