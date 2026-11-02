<div align="center">

# AnyMo
### Scaling Any-Modality Conditional Motion Generation with Masked Modeling

Yiheng Li · Zhuo Li · Ruibing Hou · Yingjie Chen · Hong Chang · Hao Liu · Shiguang Shan

[![arXiv](https://img.shields.io/badge/arXiv-2605.29488-b31b1b.svg)](https://arxiv.org/abs/2605.29488)
[![Project](https://img.shields.io/badge/Project-Page-2563eb)](https://interestingzhuo.github.io/anymo-page/)
[![Models](https://img.shields.io/badge/Hugging_Face-Models-ffd21e)](https://huggingface.co/human-motion-capture/AnyMo)
[![Dataset](https://img.shields.io/badge/Hugging_Face-OmniHuMo-ffd21e)](https://huggingface.co/datasets/L-yiheng/OmniHuMo)

[Project Page](https://interestingzhuo.github.io/anymo-page/) · [arXiv](https://arxiv.org/abs/2605.29488) · [Hugging Face Models](https://huggingface.co/human-motion-capture/AnyMo) · [OmniHuMo Dataset](https://huggingface.co/datasets/L-yiheng/OmniHuMo) · [Processed Test Data](https://huggingface.co/datasets/human-motion-capture/processed_test_dataset_of_AnyMo)

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

AnyMo generates human motion from text, speech, music, trajectories, and combinations of these signals. It pairs a residual finite scalar quantization (R-FSQ) motion tokenizer with a masked Transformer. The accompanying OmniHuMo dataset contains 3.2 million motion sequences and 5,048 hours of motion.

This release candidate provides a standalone **text / trajectory inference path for the released Stage3 3B checkpoint**, plus the original training and evaluation implementations. Speech/music feature extraction and full training/evaluation have not been validated as part of this packaging pass. See [release status](docs/RELEASE_STATUS.md).

## Installation

The verified inference environment uses **Linux, Python 3.11, PyTorch 2.5.1, and CUDA**. The original dependency snapshot used CUDA 12.1; reproduction also ran successfully with PyTorch 2.5.1 + CUDA 12.4 on an NVIDIA H20. CPU-only and Apple MPS inference are not supported by this entry point.

From this repository's root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Plan for a GPU with 24 GB or more VRAM and sufficient host RAM; 64 GB host RAM is a practical starting point for loading the full 3B checkpoint. Observed GPU allocation was approximately 17 GiB on H20, not a hardware-independent minimum. The full checkpoint is approximately 17.6 GB and includes frozen T5 and motion-tokenizer parameters.

## Assets and checkpoints

The code, checkpoints, and datasets are separate artifacts. If a Hugging Face repository requires access, first obtain access from its owner and run `hf auth login` with a read-enabled token. Do not put tokens in scripts or repository files.

Download the small normalization and T5 tokenizer/configuration files:

```bash
python tools/download_assets.py
```

This does **not** download another copy of the T5 weights: the released full AnyMo checkpoint already contains them. The statistics come from the [processed test dataset](https://huggingface.co/datasets/human-motion-capture/processed_test_dataset_of_AnyMo), pinned to the verified dataset revision.

Download the model archives, then concatenate parts in filename order before extraction:

```bash
hf download human-motion-capture/AnyMo \
  --revision 8b3f5719d4b0a0084c09a1c038266c511e6cb09f \
  --include 'work_dirs.tar.gz.part_*' --local-dir downloads/models
cat downloads/models/work_dirs.tar.gz.part_* | tar -xz
```

The archived model bundle is approximately 57.3 GB compressed. Leave space for both the archives and their extracted contents. Expected files for the documented inference path:

```text
checkpoints/flan-t5-xl/
  config.json
  spiece.model
  tokenizer_config.json
  special_tokens_map.json
data/
  mean_272.npy
  std_272.npy
  traj_mean.npy
  traj_std.npy
work_dirs/anymo_stage_3_3B_210e_48gpu/
  lite_epoch_210.pth
  anymo_stage_3.py
```

The supplied `configs/inference/anymo_3b.py` matches the archived 3B architecture: 24 layers, 32 attention heads, and hidden width 3200. Use a matching archived configuration for a different model size. Loading is strict; mismatched or incomplete checkpoints are rejected. Configurations mentioning 7B are training templates, not evidence that a 7B checkpoint is included in the verified bundle.

## Text-to-motion

```bash
python tools/infer.py \
  --checkpoint work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth \
  --prompt "A person steadily walks forward, alternately stepping with the left and right foot while swinging their arms." \
  --frames 300 --seed 3407 --out outputs/walking
```

Motion is generated at **30 fps**, with **60–300 frames in multiples of four**. Specific descriptions of body movements often work better than short generic prompts. The native sampler defaults to 6 iterations and `--topk-threshold 0.9`, which retains the top 10% of logits; it is not nucleus/top-p sampling.

## Text + trajectory

Provide an unnormalized pelvis trajectory as a NumPy `.npy` array of shape `(T, 3)`, or an `.npz` containing `target_traj`. Coordinates are world **XYZ in meters**, with **Y up**, sampled at 30 fps. Keep the same origin and orientation convention as the motion representation. The program applies the published trajectory normalization exactly once. It does not silently resample, rotate, align, or scale the supplied path.

```bash
python tools/infer.py \
  --checkpoint work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth \
  --prompt "A person turns their body while swinging both arms." \
  --trajectory path/to/trajectory.npy \
  --tta-steps 400 --tta-lr 0.0002 --out outputs/trajectory
```

Omit `--prompt` for trajectory-only generation. Omit `--tta-steps` for native conditional generation. `tools/test_traj.py` is an alias requiring a trajectory and defaulting to 400 fitting steps.

**Trajectory fitting updates decoder latents, not model weights.** It minimizes pelvis XYZ MSE using AdamW (`lr=2e-4`, `betas=(0.5, 0.9)`, `weight_decay=1e-6`). This release removes in-place operations that previously broke its backward pass. Native conditioning remains a soft constraint; fitting improves tracking on some inputs but does not guarantee natural foot contact or success on arbitrary paths. Larger learning rates can diverge.

## Outputs

Each run saves `native.npz`, `report.json`, and `resolved_config.py`. Successful fitting also saves `optimized.npz`. The original sampled tokens in the optimized file describe the initialization; optimized continuous latents are not re-quantized into new tokens.

| NPZ field | Meaning |
|---|---|
| `normalized` | Model output, `(T, 272)` |
| `motion` | Denormalized motion representation, `(T, 272)` |
| `joints` | Recovered 22-joint XYZ skeleton, `(T, 22, 3)`, meters |
| `tokens` | Sampled residual motion codes |
| `prompt`, `fps` | Input text and frame rate |
| `target_traj` | Requested pelvis trajectory, when supplied |

`report.json` includes the seed, conditions, sampler settings, strict-load status, and absolute trajectory errors when applicable. Failed nonfinite fitting stops explicitly and preserves the native output. See [data conventions](docs/DATA_FORMAT.md) and [troubleshooting](docs/TROUBLESHOOTING.md).

## Training, evaluation, and tests

Training configurations and split lists are retained under `configs/` and `configs/data_paths/`. Prepare the complete training data and evaluator assets before using them; downloading the processed test subset is not sufficient for training. [Training notes](docs/TRAINING.md) describe the remaining setup.

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

The unit tests exercise coordinate recovery and the trajectory-gradient regression. A single-motion, three-seed reproduction check is documented in [validation notes](docs/VALIDATION.md); it is not a replacement for paper benchmark results.

The newly packaged CLI still needs a fresh GPU integration run. The CPU tests and source checks passed, while the remote GPU recheck was blocked by an automatic permission-review timeout. Historical reproduction numbers are labeled separately in the validation notes.

## Citation

```bibtex
@misc{li2026anymoscalinganymodalityconditional,
  title={AnyMo: Scaling Any-Modality Conditional Motion Generation with Masked Modeling},
  author={Yiheng Li and Zhuo Li and Ruibing Hou and Yingjie Chen and Hong Chang and Hao Liu and Shiguang Shan},
  year={2026},
  eprint={2605.29488},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.29488}
}
```

## License and acknowledgments

The authors have not yet selected a code license for this release candidate; see [license status](LICENSE_STATUS.md). No code license is implied by the dataset license or the paper's arXiv distribution terms. Dataset, pretrained-model, and third-party implementation terms must be handled separately. Existing attribution is preserved; see [third-party notices](THIRD_PARTY_NOTICES.md).
