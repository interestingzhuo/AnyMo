# Model checkpoints

Download the released model bundle from **[AnyMo on Hugging Face](https://huggingface.co/human-motion-capture/AnyMo)**.

The checkpoint files below are included inside `work_dirs.tar.gz.part_*`. Download and extract the bundle to obtain them. The Hugging Face repository's visibility controls access; a public repository can be downloaded without logging in, while a private repository requires access and `hf auth login`.

## Included checkpoints

| Component | Extracted checkpoint | Size | Purpose |
|---|---|---:|---|
| Stage3 3B | `work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth` | 17.61 GB | Text, audio and trajectory motion model; the documented inference CLI supports text/trajectory |
| Stage2 3B | `work_dirs/anymo_stage_2_3B_210e_48gpu/lite_epoch_210.pth` | 17.61 GB | Initialization for Stage3 training |
| Motion tokenizer | `work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth` | 94.79 MB | Stage1 tokenizer initialization and reconstruction evaluation |
| TEMOS evaluator | `work_dirs/temos_50e/epoch_50.pth` | 436.37 MB | Text-motion evaluation; requires additional encoder assets |

Sizes use decimal GB/MB. The full AnyMo checkpoints contain frozen T5 and motion-tokenizer weights. The tokenizer and evaluator are separate checkpoints with different loading interfaces. The bundle also includes the corresponding archived configurations.

## Download and extract

Run from the code repository root:

```bash
hf download human-motion-capture/AnyMo \
  --revision 8b3f5719d4b0a0084c09a1c038266c511e6cb09f \
  --include 'work_dirs.tar.gz.part_*' --local-dir downloads/models
cat downloads/models/work_dirs.tar.gz.part_* | tar -xz
python tools/download_assets.py
```

The pinned revision selects the original released bundle. Download every part and concatenate them in filename order. The compressed model bundle is approximately 57.3 GB; reserve additional space for the extracted files. `tools/download_assets.py` downloads normalization arrays and T5 tokenizer/configuration files, not model weights.

Individual `.pth` files are currently distributed inside this bundle. The paths in the table describe the extracted layout; they are not separate Hugging Face file-download URLs.

## Use the matching interface

- **Inference:** use Stage3 3B with `configs/inference/anymo_3b.py` and `tools/infer.py`; see the root [README](../README.md).
- **Training:** use the trained tokenizer for Stage1, your Stage1 checkpoint for Stage2, and Stage2 for Stage3; see [Train](TRAINING.md). The 7B parent templates do not imply that the bundle includes a 7B checkpoint.
- **Evaluation:** use the model/tokenizer checkpoint with the matching evaluator configuration. TEMOS additionally requires the external encoder assets documented in [Evaluation](EVALUATION.md).

The source-code license does not replace the terms for model weights, pretrained encoders or datasets. Refer to the Hugging Face repository and [licensing scope](../LICENSE_STATUS.md).
