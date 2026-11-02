# Reproducing the full evaluation

The portable entry point is `tools/evaluate.py`. It preserves the released datasets, random caption/crop logic, sampler, feature encoders and metric functions used in the 2026-11-02 full run. It runs on one CUDA GPU; it does not launch infrastructure-specific GPU occupancy processes.

## Assets

Use Python 3.11 and the PyTorch 2.5.1 CUDA environment from the root README:

```bash
python -m pip install -r requirements/benchmark.txt
python tools/download_assets.py
```

Download/extract the pinned model bundle as documented in the README. Required files:

```text
work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth
work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth
work_dirs/temos_50e/temos.py
work_dirs/temos_50e/epoch_50.pth
checkpoints/flan-t5-xl/                 # downloaded by download_assets.py
```

Prepare the official test data and DistilBERT tokenizer/configuration:

```bash
python tools/prepare_evaluation.py
```

The helper downloads the seven processed-test shards at dataset revision `8baf0d5da5f5dcc44a1281f2d251dbaa04118209`, approximately 59.8 GB compressed. Allow additional space for the 11,415 extracted test files and evaluation caches. It checks every source shard against its Hugging Face LFS SHA256, extracts only regular files named in the published test lists, and writes `data/evaluation_manifest.json`. It does not redistribute test motions in Git.

For an existing extracted dataset:

```bash
python tools/prepare_evaluation.py --backbone-only
python tools/evaluate.py --data-root /path/to/test-data --check-only
```

`--data-root` must contain `omnihumo/`, not point inside it. The backbone helper fetches only configuration/tokenizer files pinned to `12040accade4e8a0f71eabdb258fecc2e7e948be`. Learned DistilBERT weights come from the published TEMOS checkpoint. Only its text and motion encoders are strictly loaded; the decoder, sentence-transformer training filter and losses are not used for these metrics. `sentence-transformers` is not needed for this entry point.

## Run

From the repository root:

```bash
python tools/evaluate.py --check-only
python tools/evaluate.py --out outputs/full_evaluation
```

For assets stored separately:

```bash
python tools/evaluate.py \
  --assets-dir /path/to/model-assets \
  --data-root /path/to/test-data \
  --out outputs/full_evaluation
```

All paths have explicit overrides: `--checkpoint`, `--config`, `--evaluator-config`, `--evaluator-checkpoint`, `--tokenizer-checkpoint`, `--text-backbone`, `--split-dir`. Run `python tools/evaluate.py --help` without GPU dependencies to see them.

To run only part of the benchmark:

```bash
python tools/evaluate.py --phases text text_traj --out outputs/generation
python tools/evaluate.py --phases reconstruction --out outputs/reconstruction
```

A short integration check:

```bash
python tools/evaluate.py --max-batches 1 --out outputs/benchmark_smoke
```

Limited runs are explicitly marked `partial` and are not paper benchmark results. A single batch gives a statistically meaningless FID; its purpose is verifying the pipeline. Use a different output directory for the full run.

## Resume and outputs

Re-run the same command to resume. Completed batches and phase results are reused. The run signature hashes model/evaluator/tokenizer files, normalization arrays, text-backbone assets, source and split lists; incompatible signatures are rejected. Each cached batch also stores its input digest and the CUDA RNG state after generation. Resume still reads preceding dataset batches to reproduce worker crop/caption state. A new process must keep the same dataset contents, dependencies, seed and CUDA environment. Test data mutation is detected for cached inputs, but the helper's manifest remains the source of archive integrity evidence.

Outputs include `results.json`, `protocol.json`, `run_signature.json`, `progress.json`, `REPORT.md`, and one NPZ per completed batch. Batch NPZs store evaluator features, error statistics, lengths, sampled token IDs, captions, input hashes and RNG state. No optimizer updates or test-time fitting are applied to these benchmark predictions.

The complete single-seed reference took about 2.7 hours for text, 3.8 hours for text + trajectory and 3.7 minutes for reconstruction on one H20. These are measured run times, not hardware-independent guarantees. Loading used about 18–24 GB GPU memory in this setup.

## Exact protocol and comparison limits

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
