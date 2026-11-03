# Evaluation

Install research dependencies and run from the repository root:

```bash
python -m pip install -r requirements/evaluation.txt
```

## Data and checkpoints

See [Data preparation](DATA_PREPARATION.md) for the pinned test-data download and expected extraction layout.

Download the model, tokenizer and TEMOS checkpoints from the [Hugging Face model repository](https://huggingface.co/human-motion-capture/AnyMo); see [Model checkpoints](CHECKPOINTS.md) for the bundle layout.

Extract the [processed-test dataset](https://huggingface.co/datasets/human-motion-capture/processed_test_dataset_of_AnyMo) to `data/omnihumo/`, concatenating archive parts in filename order before extraction. Place the published normalization arrays under `data/`.

The model bundle described in the README contains:

```text
work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth
work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth
work_dirs/temos_50e/temos.py
work_dirs/temos_50e/epoch_50.pth
```

Text evaluation also needs the encoder assets referenced by TEMOS:

```bash
hf download distilbert/distilbert-base-uncased --local-dir checkpoints/distilbert-base-uncased \
  --include 'config.json' 'tokenizer*' 'vocab.txt' 'special_tokens_map.json' 'model.safetensors'
hf download sentence-transformers/paraphrase-MiniLM-L6-v2 --local-dir checkpoints/sentence-transformers-paraphrase-MiniLM-L6-v2 \
  --include 'config*.json' 'tokenizer*' 'vocab.txt' 'special_tokens_map.json' \
  'model.safetensors' 'modules.json' 'sentence_bert_config.json' '1_Pooling/config.json'
```

Set `model.text_encoder_cfg.modelpath` in `work_dirs/temos_50e/temos.py` to `checkpoints/distilbert-base-uncased` if needed. The sentence-transformer directory must use the name above. `tools/download_assets.py` supplies motion statistics and FLAN-T5 configuration/tokenizer files; it does not supply every evaluator asset.

`configs/eval_text.py` allocates T5 from local configuration and loads its weights from the full AnyMo checkpoint. Evaluation does not require another copy of pretrained T5 weights. Stage1 training does require pretrained T5 weights; see [Train](TRAINING.md).

## Text + trajectory

This is the default condition in `configs/eval_text.py`:

```bash
python tools/test.py configs/eval_text.py \
  work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth \
  --work-dir work_dirs/eval_text_traj
```

## Text only

Override both the model condition and evaluator:

```bash
python tools/test.py configs/eval_text.py \
  work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth \
  --work-dir work_dirs/eval_text \
  --cfg-options model.cond_types="['text']" test_evaluator.type=Text2MotionMetric
```

The default loader reads 2 pickle files per batch, each of which can contain multiple motion records. Retrieval negatives come from the expanded batch. Use enough test motions for FID and Diversity; fewer than 101 motions cannot satisfy the original small-set Diversity sampling count.

## Tokenizer reconstruction

```bash
python tools/test.py configs/tokenizer_r_fsq.py \
  work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth \
  --work-dir work_dirs/eval_tokenizer
```

This uses `ReconsMetric`; text generation uses the TEMOS feature encoders. Their asset requirements differ.

## Distributed evaluation

```bash
bash tools/dist_test.sh configs/eval_text.py \
  work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth 8 \
  --work-dir work_dirs/eval_text_traj
```

The third positional argument is GPU count; `PORT` selects the process-group port. Match the architecture, normalization and dataset paths to the checkpoint before starting the runner.
