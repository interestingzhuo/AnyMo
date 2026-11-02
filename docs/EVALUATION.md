# Evaluation notes

The original research entry points are `tools/test.py` and `tools/dist_test.sh`. Configurations are under `configs/`, including `eval_text.py`, `eval_speak.py`, `eval_dance.py` and `tokenizer_r_fsq.py`.

## Required assets and original interfaces

Prepare the data tree under `data/`, the published normalization arrays, matching model checkpoint/configuration and the TEMOS config/checkpoint. Original research module registration may also instantiate sentence-transformer training/evaluator assets; install the research dependencies from `requirements/evaluation.txt` and prepare the encoder assets referenced by each config. The minimal inference download helper does not prepare every asset needed by the original research runner.

```bash
# Original interface; inspect the selected config and prepare its assets first.
python tools/test.py CONFIG CHECKPOINT --work-dir work_dirs/evaluation
bash tools/dist_test.sh CONFIG CHECKPOINT 1
```

`configs/eval_text.py` defaults to the 3B text+trajectory condition. Select the intended conditions and evaluator explicitly for text-only evaluation.

The processed-test dataset is available from [Hugging Face](https://huggingface.co/datasets/human-motion-capture/processed_test_dataset_of_AnyMo). Concatenate its archive parts in filename order and extract the `omnihumo/` tree under `data/`.

Checkpoint paths used by the 3B evaluation configuration:

```text
work_dirs/anymo_stage_3_3B_210e_48gpu/lite_epoch_210.pth
work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth
work_dirs/temos_50e/temos.py
work_dirs/temos_50e/epoch_50.pth
```
