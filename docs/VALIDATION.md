# Validation evidence

## Full default evaluation — completed 2026-11-02

Reference code commit: `397447f21ff23e5efaad7a48c44159c48b42608d`. Environment: Linux, Python 3.11, PyTorch 2.5.1+cu124, NVIDIA H20; FP32, seed3407, native six-step independent-mask sampling, top-k filter threshold0.9.

- Text generation: 51,408 motions, FID55.6296, text→motion R@1 75.1167%.
- Text + trajectory: 51,408 motions, FID41.1119, R@1 77.0386%, average path error27.3120cm.
- Tokenizer reconstruction: 182,640 motions /11,415 files; root-relative MPJPE13.2649mm.
- Standalone tokenizer state equals the embedded tokenizer state.
- First reconstruction batch matches the original NumPy recovery + `ReconsMetric` to maximum0.0000153mm.
- Native regression against the saved single-clip output: token equality and normalized maximum difference0.0.
- Sampled codes outside FSQ[0,1919]: zero in both complete generation runs.

[Full protocol and metrics](../validation/full_20261102/results.json) · [Paper comparison](../validation/full_20261102/REPORT.md) · [Commands and limitations](EVALUATION.md).

The paper's historical launcher was unavailable; the author confirmed using public defaults. The final retrieval group has16 candidates rather than32. Both retrieval directions are saved. Diversity pair selection is seeded separately, so exact archived Diversity is not expected. This is not a complete training/audio reproduction.

## Small-subset trajectory optimization

A separate512-motion, three-seed diagnostic used the corrected canonical recovery. On the same32 motions per seed,400-step latent fitting reduced ADE from29.906±1.531cm to4.429±1.233cm; FDE from60.510±5.208cm to6.480±2.000cm. All optimizations had finite losses and gradients. This uses a known cropped GT target path and is not the unoptimized full benchmark above. It does not establish foot-contact quality on arbitrary trajectories.

The earlier single selected-motion fitting result used an older helper with a different treatment of non-yaw rotations; it is superseded by these canonical-recovery checks and should not be advertised as a current release result.

## Repository tests

The retained tests cover CLI help/invalid parameters and coordinate-recovery forward/gradient regressions. These four tests passed in the PyTorch environment. Source/link checks found no technical issues. Strict publication checks intentionally report the retained SnapMoGen non-commercial condition.

The full benchmark evidence above remains valid. The original research entry points remain `tools/test.py` and `tools/dist_test.sh`. No full training or audio benchmark claim is made here.
