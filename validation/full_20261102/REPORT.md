# AnyMo 3B — complete default evaluation

Single H20, seed3407, FP32, six refinement steps, published test lists and dataset/collate/metric functions.
Public eval_text.py defaults: DefaultSampler shuffle=True; 2 files per batch; 4 data workers. Tokenizer uses its complete default test list, 4 files per batch, 8 workers, no shuffle.
Historical multi-GPU launcher is unavailable; the author confirmed use of public defaults. This run contains no distributed-padding duplicate files. Final smaller retrieval groups are disclosed.

| Condition / metric | Archived paper result | New full test | Difference |
|---|---:|---:|---:|
| text: FID | 55.5974 | 55.6296 | +0.0322 |
| text: R@1 text→motion (%) | 75.0019 | 75.1167 | +0.1148 |
| text: R@2 text→motion (%) | 86.8412 | 86.6130 | -0.2282 |
| text: R@3 text→motion (%) | 91.3635 | 91.2523 | -0.1112 |
| text: MMDist | 25.8787 | 25.8646 | -0.0140 |
| text: Diversity | 46.7124 | 46.1759 | -0.5365 |
| text: Mean path error (cm) | 50.5060 | 50.6992 | +0.1933 |
| text: Trajectory fail >50cm (%) | 52.1105 | 52.3634 | +0.2529 |
| text: Frame fail >50cm (%) | 28.4647 | 28.5597 | +0.0950 |

text samples: 51408; files: 3213; candidate group sizes: {'32': 1606, '16': 1}.
Real calibration: R@1 74.3581%; MMDist 25.753432; Diversity 46.809723.
Paper-described motion→text R@1/2/3: [74.6479, 86.7783, 91.3924]%.


| Condition / metric | Archived paper result | New full test | Difference |
|---|---:|---:|---:|
| text_traj: FID | 41.4394 | 41.1119 | -0.3275 |
| text_traj: R@1 text→motion (%) | 76.9881 | 77.0386 | +0.0505 |
| text_traj: R@2 text→motion (%) | 88.8312 | 89.0075 | +0.1763 |
| text_traj: R@3 text→motion (%) | 93.0892 | 93.3084 | +0.2192 |
| text_traj: MMDist | 25.3841 | 25.3791 | -0.0050 |
| text_traj: Diversity | 46.7124 | 46.3553 | -0.3571 |
| text_traj: Mean path error (cm) | 27.1594 | 27.3120 | +0.1526 |
| text_traj: Trajectory fail >50cm (%) | 32.9913 | 33.0746 | +0.0833 |
| text_traj: Frame fail >50cm (%) | 13.7383 | 13.8238 | +0.0855 |

text_traj samples: 51408; files: 3213; candidate group sizes: {'32': 1606, '16': 1}.
Real calibration: R@1 74.3581%; MMDist 25.753432; Diversity 46.809723.
Paper-described motion→text R@1/2/3: [76.5562, 88.4726, 92.8999]%.

## Tokenizer reconstruction

Full default list: 182640 motions, 11415 files. Per-motion mean root-relative MPJPE: **13.2649 mm**; paper Table4 OmniHuMo: 13.2 mm.
Standalone tokenizer weights were compared with the embedded tokenizer. The accelerated metric was checked against the actual released NumPy recovery + ReconsMetric on the first batch.

## FID versus sample count (same cached predictions)

| Samples | Text FID | Text + trajectory FID |
|---:|---:|---:|
| 512 | 101.8162 | 87.3626 |
| 1024 | 80.7540 | 64.8512 |
| 2048 | 67.4543 | 52.9027 |
| 4096 | 59.3918 | 47.0033 |
| 8192 | 57.5123 | 44.8072 |
| 16384 | 55.5824 | 41.9516 |
| 32768 | 56.5221 | 41.7165 |
| 51408 | 55.6296 | 41.1119 |

These are nested prefixes of a single run, not independent repeated trials. They diagnose sample-count sensitivity; they do not remove FID bias.

The archived trajectory-error fields are fractions despite the paper table percent labels. This report converts them to percentages explicitly.

Paper: https://arxiv.org/html/2605.29488v3

Status: complete. Full protocol, split hashes and metrics: [results.json](results.json). Original reference fields: [paper_reference.json](paper_reference.json). Individual generated motions and batch caches are not redistributed here. Protocol and original research entry points: [evaluation notes](../../docs/EVALUATION.md).
