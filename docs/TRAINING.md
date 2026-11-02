# Training and evaluation

The original model, dataset, loss and metric implementations are retained. The standalone inference config is intentionally separate from the training templates.

1. Prepare the complete OmniHuMo data, motion normalization and required pretrained encoders.
2. Install the additional research dependencies with `python -m pip install -r requirements/evaluation.txt`. Distributed/deepspeed launchers additionally require a compatible DeepSpeed installation, GPU interconnect configuration and the original training hardware setup.
3. Inspect the chosen config's model size, initialization checkpoint and dataset paths. Historical configurations may refer to intermediate checkpoints that are not included in the published model bundle.
4. Keep split lists under `configs/data_paths/`; place the data tree under `data/` or update paths explicitly.

```bash
python tools/train.py configs/tokenizer_r_fsq.py --work-dir work_dirs/tokenizer
bash tools/dist_train.sh configs/anymo_stage_3.py 8 --work-dir work_dirs/stage3
```

The Stage3 training template currently defines 7B; select the intended model size and a matching Stage2 checkpoint before starting it.

The launchers accept `MASTER_ADDR`, `PORT`, `NNODES` and `NODE_RANK` from the environment. Configure RDMA and networking for the target cluster.

Evaluation requires evaluator checkpoints and encoders separately, for example the TEMOS config/checkpoint and sentence-transformer assets referred to by `configs/eval_*.py`. Inference does not require these evaluator weights. See [evaluation instructions](EVALUATION.md) for the research test interfaces. Model and evaluator registration is explicit through `src.register_all_modules()` in the research tools.

`requirements/legacy-freeze.txt` preserves the original dependency snapshot for comparison. It includes a CUDA-specific PyTorch build and is not the recommended portable installation command.
