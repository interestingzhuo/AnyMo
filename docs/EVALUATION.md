# Evaluation notes

The original research entry points are `tools/test.py` and `tools/dist_test.sh`. Configurations are under `configs/`, including `eval_text.py`, `eval_speak.py`, `eval_dance.py` and `tokenizer_r_fsq.py`.

## Required assets and original interfaces

Prepare the data tree under `data/`, the published normalization arrays, matching model checkpoint/configuration and the TEMOS config/checkpoint. Original research module registration may also instantiate sentence-transformer training/evaluator assets; install the research dependencies from `requirements/evaluation.txt` and prepare the encoder assets referenced by each config. The minimal inference download helper does not prepare every asset needed by the original research runner.

```bash
# Original interface; inspect the selected config and prepare its assets first.
python tools/test.py CONFIG CHECKPOINT --work-dir work_dirs/evaluation
bash tools/dist_test.sh CONFIG CHECKPOINT 1
```

`configs/eval_text.py` defaults to the 3B text+trajectory condition. Select the intended conditions and evaluator explicitly for text-only evaluation. These are the retained research interfaces, not a claim that this exact command was rerun from a fresh checkout during packaging. The full results below were obtained using the released model helpers, dataset/collate functions and metric implementations with the documented defaults.

The processed-test dataset is pinned to revision `8baf0d5da5f5dcc44a1281f2d251dbaa04118209`. Its seven archive parts total approximately 59.8 GB compressed. Concatenate them in filename order and extract the `omnihumo/` tree under `data/`. All seven source shards were SHA256-verified for the recorded run. No raw test motions are committed here.

Required checkpoint paths for the recorded run:

```text
work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth
work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth
work_dirs/temos_50e/temos.py
work_dirs/temos_50e/epoch_50.pth
```

## Recorded full-run protocol and comparison limits

- FP32, seed 3407, native six-step sampler and top-k filter threshold 0.9. Model weights and sampling logic are unchanged.
- Text: all 3,213 dynamic-caption test files, 51,408 motions; `DefaultSampler(shuffle=True)`, two files per batch and four loader workers. Each file contains 16 motions. There are 1,606 retrieval groups of 32 and one final group of 16, matching the public default loader; that last group has fewer negatives than the paper's stated 32-candidate protocol.
- Reconstruction: 11,415 files / 182,640 motions from the five test lists in `configs/tokenizer_r_fsq.py`, four files per batch, eight loader workers, no shuffle. Standalone tokenizer state must equal the embedded tokenizer state. MPJPE aligns the pelvis per frame and averages per motion. The first batch is checked against the released NumPy recovery and `ReconsMetric`.
- Feature metrics use the exact TEMOS text/motion encoder means, with strict state loading and zero padding to 300 frames. Both retrieval directions are reported: the released code computes text→motion; the paper describes motion→text.
- FID uses released sample covariance and SciPy matrix square root. Diversity samples 300 random pairs. Different pair selections can move Diversity; its goal is closeness to real diversity, not maximizing it.
- Trajectory metrics use the released NumPy recovery and strict per-sequence failure rule. JSON failure fields are fractions; multiply by 100 to express percentages. `ade_m` and `fde_m` are meters. The paper table's percentage labels do not match the raw fraction display in its archived JSON.
- Official trajectory inputs recover the full raw path before cropping. Evaluation recovers cropped GT motion; this convention is deliberately preserved.
- The authors could not recover the original paper launcher and confirmed using public defaults. Historical distributed padding / GPU count is not reconstructed. Small numerical differences are expected; this is a full default-code reproduction, not a claim of bitwise identity with a historical job.
- Full training, speech/music feature preparation and their benchmarks have not been revalidated by this run.

## Verified results

[Full run and protocol](../validation/full_20261102/results.json) · [Detailed paper comparison](../validation/full_20261102/REPORT.md)

| Metric | Archived paper result | Full default run |
|---|---:|---:|
| Text FID ↓ | 55.5974 | 55.6296 |
| Text R@1 ↑ | 75.0019% | 75.1167% |
| Text + trajectory FID ↓ | 41.4394 | 41.1119 |
| Text + trajectory R@1 ↑ | 76.9881% | 77.0386% |
| Text + trajectory ADE ↓ | 27.1594 cm | 27.3120 cm |
| Reconstruction MPJPE ↓ | 13.2 mm (paper table) | 13.2649 mm |

The first 512 cached text predictions have FID 101.8162, versus 55.6296 for all 51,408. Do not compare small-subset FID directly with the full paper table. Prefix FIDs and both retrieval directions are included in the saved results.
