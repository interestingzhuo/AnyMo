"""3B condition-encoder training initialized from Stage1."""
_base_ = ['../anymo_stage_2.py']
model_size = '3B'
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint='work_dirs/stage1_3b/epoch_210.pth'),
    transformer_cfg=dict(n_layer=24, n_head=32, n_embd=3200),
    text_encoder_cfg=dict(output_dim=3200, pretrained=False, local_files_only=True),
    traj_encoder_cfg=dict(output_emb_width=3200),
    audio_encoder_cfg=dict(width=512, output_emb_width=3200),
)
