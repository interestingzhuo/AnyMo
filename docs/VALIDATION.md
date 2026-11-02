# Reproduction observations

**Packaging status:** four CPU regression tests passed in the existing PyTorch 2.5.1 environment. A fresh end-to-end GPU run of the new standalone CLI has not completed: automatic permission review timed out twice before the remote import check could start. The historical figures below describe the earlier reproduction, not a newly executed release-CLI benchmark.

The packaging regression tests also exposed a semantic difference in the earlier autograd helper for non-yaw rotations. The release now strictly preserves the original recovery function's X/Z-only translation integration, with forward-equivalence tests on general 3D rotations. Historical latent-fitting numbers should be rechecked with this exact release implementation before being used as release-CLI results.

The preparation work used the full Stage3 3B epoch210 checkpoint, strict loading, FP32, native 6-step independent-mask sampling, top-k threshold 0.9, and seeds 3407, 42, 2026. The reference environment was Python 3.11 / PyTorch 2.5.1+cu124 / NVIDIA H20. The release entry point additionally receives a regression smoke test recorded in `validation/release_smoke.json` when that check is available.

The existing trajectory experiment used a selected 296-frame motion from the processed test dataset: `motion_caption_speak_pkl/0003533.pkl`, record 5, starting at frame 0. It contains turning, arm movement and side-to-side travel. No origin/scale alignment was applied when computing errors.

| Condition | Mean pelvis ADE (m) | Mean final error (m) |
|---|---:|---:|
| Text only |0.8152|0.4708|
| Trajectory only |0.7003|2.2748|
| Text + trajectory |0.8857|1.1215|
| Text + trajectory + 400-step latent fitting |0.1251|0.1329|

All three 400-step optimizations were finite at `lr=2e-4`. Four-limb motion variation remained similar before/after fitting. Approximate near-floor foot speed increased from 0.365 to 0.408 m/s, while the real reference's corresponding proxy was 0.297 m/s. This proxy is not a certified contact metric or a perceptual score.

The same motion's VQ reconstruction had approximately 0.134 m pelvis ADE and 0.035 m root-relative joint error. Forward joint recovery in the reproduction matched the selected NumPy reference within 4.77e-7 m.

These are observations from one selected motion, not representative dataset averages, paper benchmarks, or guarantees for arbitrary trajectories. Earlier synthetic straight/turn paths were fitted less successfully by the same defaults. Short generic text prompts also sometimes produced near-static motion.

The publication package omits experiment pickles, raw reference sequences, large weight files and generated videos. Code, weights, source data and visualization media require separately reviewed release terms.
