# Third-party implementation references

Existing source headers and attribution comments have been retained. This inventory identifies upstream references in the supplied code; it does not replace their license files.

| Area | Referenced implementation |
|---|---|
| MMEngine runner and tool scaffolding | https://github.com/open-mmlab/mmengine |
| Motion encoders and quantizer variants | https://github.com/snap-research/SnapMoGen |
| Evaluator / ACTOR modules | https://github.com/Mathux/TMR |
| RMS normalization | https://github.com/bzhangGo/rmsnorm |
| Transformer utility reference | https://github.com/labmlai/annotated_deep_learning_paper_implementations |
| 6D rotation representation | https://arxiv.org/abs/1812.07035 |
| Frozen text encoder and tokenizer | https://huggingface.co/google/flan-t5-xl |

Upstream license copies and their scope are recorded below. Attribution comments and original copyright headers must not be removed. The root Apache-2.0 license covers original AnyMo contributions, while Python dependency licenses and component restrictions remain applicable independently.

## Retained upstream license texts (checked 2026-11-02)

| Referenced code | Files / scope | Retained terms |
|---|---|---|
| SnapMoGen | `src/models/tokenizers/encdoc/encdoc_attn.py`, `src/models/tokenizers/quantizer/ms_ema_reset.py`; related motion-tokenizer adaptations require the same provenance review | [Snap Inc. Non-Commercial License](licenses/SnapMoGen.txt): non-commercial research only; not overridden by Apache-2.0 |
| TMR / ACTOR evaluator | `src/models/evaluator/` | [MIT](licenses/TMR.txt) |
| RMSNorm | RMSNorm in `src/models/transformers/llama.py` | [BSD-3-Clause](licenses/RMSNorm.txt) |
| labml RoPE reference | `build_rope_cache` in `src/models/transformers/llama.py` | [MIT](licenses/labml.txt) |
| PyTorch3D rotation utilities | `src/utils/rotation_conversions.py` (original Facebook copyright retained) | [BSD](licenses/PyTorch3D.txt) |
| MMEngine tool scaffolding | research runner / training and test tool adaptations | [Apache-2.0](licenses/MMEngine.txt) |
| Facebook motion feature utilities | `src/utils/audio_feat/{utils,kinetic,manual,manual_new}.py` | Original BSD license text is already embedded at the top of each source file and is retained |

The upstream references identify provenance, not a representation that all third-party code can be relicensed. SnapMoGen's current upstream license expressly preserves its non-commercial restrictions. Existing headers must remain intact. The original AnyMo contributions are Apache-2.0; rights for the combined code depend on all applicable component terms.
