# Train

Run commands from the repository root in the CUDA environment described in the README:

```bash
python -m pip install -r requirements/evaluation.txt
```

## Data and pretrained assets

Follow [Data preparation](DATA_PREPARATION.md) for dataset downloads, processed record formats and split-list configuration. In particular, set Stage1's currently empty training list before launching it.

Download initialization weights from the [Hugging Face model repository](https://huggingface.co/human-motion-capture/AnyMo); the [checkpoint inventory](CHECKPOINTS.md) lists their extracted paths and purposes.

- Place the complete training data under `data/omnihumo/`; keep split lists under `configs/data_paths/`. Processed test data is not a replacement for training data.
- Put `mean_272.npy`, `std_272.npy`, `traj_mean.npy` and `traj_std.npy` under `data/`.
- Audio training records need the precomputed `audio_feat` field (512 channels, 75 fps). Motion records use 272 channels at 30 fps. See [data format](DATA_FORMAT.md).
- For Stage1, download the **full** FLAN-T5-XL encoder and tokenizer to `checkpoints/flan-t5-xl/`. `tools/download_assets.py` downloads configuration/tokenizer files for inference only; it does not download these pretrained training weights.
- Tokenizer validation in `configs/tokenizer_r_fsq.py` references `data/motionmillion/mini_test.pkl`. Supply that validation set, or replace/disable the validation loader and evaluator consistently for your dataset.

For example, obtain the full T5 assets with the Hugging Face CLI:

```bash
hf download google/flan-t5-xl --local-dir checkpoints/flan-t5-xl \
  --include 'config.json' 'spiece.model' 'tokenizer*' 'special_tokens_map.json' \
  'model*.safetensors' 'model.safetensors.index.json'
```

Check dataset paths, initialization checkpoints, batch sizes and optimizer settings in the selected configuration before launching a run. Batch size counts pickle files; each file can expand to multiple motion records.

## Training stages

| Stage | Configuration | Initialization / behavior |
|---|---|---|
| Motion tokenizer | `configs/tokenizer_r_fsq.py` | Train motion encoder, residual FSQ and decoder |
| Stage1 | `configs/train/anymo_stage_1_3b.py` | Load trained tokenizer and pretrained T5; train text-to-motion model |
| Stage2 | `configs/train/anymo_stage_2_3b.py` | Load Stage1; freeze motion Transformer and text encoder; train condition encoders |
| Stage3 | `configs/train/anymo_stage_3_3b.py` | Load Stage2; joint multimodal training with frozen tokenizer/T5 |

The 3B configurations use 24 layers, 32 heads, 3200 hidden channels and 512 audio-encoder width. The parent `anymo_stage_*.py` files remain 7B templates; changing their `model_size` value via `--cfg-options` alone does not rebuild dictionaries that Python already evaluated. Use the matching model configuration.

```bash
# Tokenizer
bash tools/dist_train.sh configs/tokenizer_r_fsq.py 8 \
  --work-dir work_dirs/tokenizer_r_fsq_2048cd_4res_200e

# Stage1
bash tools/dist_train.sh configs/train/anymo_stage_1_3b.py 8 \
  --work-dir work_dirs/stage1_3b \
  --cfg-options \
  train_dataloader.dataset.data_path_file="['configs/data_paths/omnihumo_motion_caption_dynamic.txt','configs/data_paths/omnihumo_motion_caption_static.txt']"

# Stage2: point model.init_cfg.checkpoint to your Stage1 checkpoint.
bash tools/dist_train.sh configs/train/anymo_stage_2_3b.py 8 \
  --work-dir work_dirs/stage2_3b

# Stage3: defaults to the released full 3B Stage2 checkpoint.
bash tools/dist_train.sh configs/train/anymo_stage_3_3b.py 8 \
  --work-dir work_dirs/stage3_3b
```

Checkpoint names follow the configured save interval. If `epoch_210.pth` was not saved in your Stage1 run, select an existing checkpoint explicitly. To use your own Stage2 initialization:

```bash
bash tools/dist_train.sh configs/train/anymo_stage_3_3b.py 8 \
  --work-dir work_dirs/stage3_3b \
  --cfg-options model.init_cfg.checkpoint=work_dirs/stage2_3b/epoch_210.pth
```

For one GPU use `python tools/train.py CONFIG --work-dir WORK_DIR`, subject to model memory requirements. Eight GPUs above illustrate the launcher interface; select device count and batch size for your hardware. FLAN-T5 remains frozen, but Stage1 still needs pretrained weights; Stage2/3 load its weights from their full initialization checkpoints.

## Resume and distributed settings

```bash
python tools/train.py CONFIG --work-dir WORK_DIR --resume auto
python tools/train.py CONFIG --work-dir WORK_DIR --resume path/to/checkpoint.pth
```

`tools/dist_train.sh` accepts `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `PORT`. Configure RDMA/networking for the target cluster. For evaluation assets and commands, see [Evaluation](EVALUATION.md).
