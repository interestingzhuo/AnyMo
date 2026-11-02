"""3B text-conditioned training with the released tokenizer architecture."""
_base_ = ['../anymo_stage_1.py']
model_size = '3B'
model = dict(
    transformer_cfg=dict(n_layer=24, n_head=32, n_embd=3200),
    text_encoder_cfg=dict(output_dim=3200),
)
