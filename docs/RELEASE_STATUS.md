# Release status (2026-11-02)

Completed:

- English/Chinese README, project/paper/model/dataset links, citation and data conventions.
- Portable text/trajectory inference, strict full checkpoint loading and differentiable recovery.
- Full default single-H20 benchmark: 51,408 motions per generation condition and 182,640 tokenizer reconstruction motions. [Results](../validation/full_20261102/REPORT.md).
- Parameterized evaluation command, pinned test-data preparation, isolated official feature encoders, explicit retrieval directions and failure units, model/source/input cache validation and batch RNG restoration.
- Apache-2.0 selected by the authors for original contributions; upstream license texts and scope notices retained.
- Actual project homepage Code link published to the GitHub repository.

Remaining scope and release limitations:

- **Third-party restriction:** referenced SnapMoGen code remains under its non-commercial research license. Additional permission or provenance work is required before claiming unrestricted commercial use of the whole repository. The author's Apache selection does not override upstream terms.
- Model, dataset, source video and media terms remain separate. Verify intended access settings when announcing availability.
- Training and speech/music feature preparation/benchmarks have not been revalidated by the full text run.
- Historical paper test launcher could not be recovered. This run uses public single-GPU defaults; candidate groups and actual sample counts are preserved, not guessed.
- The full benchmark used the published model helpers and original metric functions. The portable wrapper is checked independently; see [validation](VALIDATION.md) for smoke results and limits.

`tools/check_release.py` checks technical issues; `--strict-publication` also reports the retained SnapMoGen restriction. An Apache LICENSE file alone does not clear third-party terms.
