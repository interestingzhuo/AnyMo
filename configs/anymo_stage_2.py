_base_=['./default_runtime.py']
nb_code = 2048
unit_length=4
audio_unit_length=8
find_unused_parameters=True
model_size = '7B' 
mean_path='data/mean_272.npy'
std_path='data/std_272.npy'
traj_mean_path='data/traj_mean.npy'
traj_std_path='data/traj_std.npy'

vqvae_cfg = dict(
    type='VQVAE', 
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='work_dirs/tokenizer_r_fsq_2048cd_4res_200e/epoch_200.pth'),
    decoder_cfg=dict(
        activation='relu',
        depth=3, dilation_growth_rate=3,
        down_t=2, input_emb_width=272,
        norm=None, output_emb_width=512,
        type='DecoderAttn', width=512),
    encoder_cfg=dict(
        activation='relu',
        depth=3, dilation_growth_rate=3,
        down_t=2, input_emb_width=272,
        norm=None, output_emb_width=512,
        stride_t=2, type='EncoderAttn', width=512),
    quantizer_cfg=dict(
        dim=512,
        levels=[8, 8, 6, 5],
        num_quantizers=4,
        quantize_dropout_prob=0.2,
        type='ResidualFSQ'),
)

llama_configs = {
    "70M": dict(n_layer=8, n_head=8, n_embd=512),
    "111M": dict(n_layer=12, n_head=12, n_embd=768),
    "343M": dict(n_layer=24, n_head=16, n_embd=1024),
    "775M": dict(n_layer=36, n_head=20, n_embd=1280),
    "1B": dict(n_layer=48, n_head=24, n_embd=1536),
    "3B": dict(n_layer=24, n_head=32, n_embd=3200),
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
}

# model
model=dict(
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='work_dirs/anymo_stage_1_7B_210e_40gpu/anymo_stage_1_7B_210e_40gpu.pth'),
    type='AnyMo', 
    transformer_cfg=dict(
        type='LLaMAHF', 
        block_size=1024, # max seq length
        vocab_size=nb_code + 2, # append pad token and mask token
        lm_head_cnt=vqvae_cfg['quantizer_cfg']['num_quantizers'],
        **llama_configs[model_size], 
    ), 
    vqvae_cfg=vqvae_cfg, 
    text_encoder_cfg = dict(
        type='TextEncoder', 
        text_encoder_name = 'flan-t5-xl', 
        max_text_length=128, 
        output_dim=llama_configs[model_size]['n_embd'],
    ), 
    traj_encoder_cfg = dict(
        type='Encoder',
        input_emb_width=3,
        down_t=2, norm=None, stride_t=2, width=512, 
        activation='relu', depth=3, dilation_growth_rate=3,
        output_emb_width=llama_configs[model_size]['n_embd'],
    ),
    audio_encoder_cfg=dict(
        type='Encoder',
        input_emb_width=512,
        down_t=3, norm=None, stride_t=2, 
        # width=512, 
        width=2048, 
        activation='relu', depth=3, dilation_growth_rate=3,
        output_emb_width=llama_configs[model_size]['n_embd'],
    ),
    max_motion_len=300, 
    motion_downsample_ratio=unit_length, 
    audio_downsample_ratio=audio_unit_length, 
    stage=2, 
)

# =============================================
# dataset
# =============================================

train_dataloader=dict(
    batch_size=1, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True, 
    sampler=dict(type='DefaultSampler', shuffle=True), 
    collate_fn=dict(type='anymo_collate_fn'), 
    dataset=dict(
        type='AnyMoDataset',
        data_path_file=[
            'configs/data_paths/omnihumo_motion_caption_speak.txt', 
            'configs/data_paths/omnihumo_motion_caption_dance.txt', 
        ], 
        unit_length=unit_length, 
        mean_path=mean_path,
        std_path=std_path,
        traj_mean_path=traj_mean_path, 
        traj_std_path=traj_std_path, 
        audio_unit_length=audio_unit_length, 
        min_motion_length=60, 
        max_motion_length=300,
    ), 
)


val_cfg=test_cfg = None

# optimizer
optim_wrapper=dict(
    type='AmpOptimWrapper',
    # dtype='float32',
    dtype='bfloat16', 
    optimizer=dict(
        type='AdamW', 
        lr=2e-4, 
        betas=(0.9, 0.99), weight_decay=0.0))

# training schedule
param_scheduler=[
    dict(type='LinearLR', by_epoch=False, start_factor=1e-5, begin=0, end=500), 
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=80, eta_min=1e-5),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=1e8)

default_hooks=dict(
    checkpoint=dict(type='CheckpointHook', max_keep_ckpts=2, interval=30), 
    logger=dict(type='LoggerHook', interval=50),
)
