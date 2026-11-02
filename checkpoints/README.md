# Text encoder configuration

Run `python tools/download_assets.py` to create `flan-t5-xl/` with its configuration and tokenizer files. Full released AnyMo checkpoints already contain frozen T5 weights, so the inference loader allocates the architecture from configuration and then strictly loads those embedded weights.
