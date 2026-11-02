# Troubleshooting

**401/404 from Hugging Face.** A repository may be private or access-controlled. Obtain read access, run `hf auth login`, and verify that the token covers the intended organization/repository. A successful account login alone does not prove repository access.

**Missing statistics.** Run `python tools/download_assets.py`, or set `--assets-dir` to an existing tree containing `data/*.npy` and `checkpoints/flan-t5-xl/`. Motion normalization and trajectory normalization are separate. Do not replace missing statistics with zeros/ones.

**`e210.pth` not found.** The verified Stage3 3B bundle contains `lite_epoch_210.pth`. The old trajectory script used a different hard-coded name; the current entry point requires `--checkpoint` explicitly.

**Checkpoint shape mismatch.** Use the matching archived model config, or the provided `configs/inference/anymo_3b.py` for the documented3B checkpoint. Do not use `strict=False` to hide architecture mismatches. Full released checkpoints include frozen T5 and tokenizer parameters; partial checkpoints are rejected.

**CUDA is unavailable.** The text encoder currently moves inputs to CUDA. CPU-only/MPS inference is not supported by this entry point.

**Trajectory optimization backward failure.** The release uses functional root-velocity construction and translation addition instead of in-place slice updates. The regression tests cover both forward equivalence and finite-difference gradients. Do not wrap the optimization loop in `torch.inference_mode()` or `torch.no_grad()`.

**Motion looks static.** Specific limb and temporal action descriptions can perform better than generic prompts. Increasing sample iterations did not consistently fix weak motion in the reproduction experiments. The release preserves the native independent-mask sampler; experimental shared-mask variants were not adopted as a default fix.

**Trajectory is ignored or the person slides.** Conditioning is soft. Use `--tta-steps 400` to fit decoder latents. Latent fitting constrains pelvis XYZ only and does not enforce contact or gait quality. Larger optimization learning rates can produce nonfinite values; the program stops and keeps `native.npz` in that case.
