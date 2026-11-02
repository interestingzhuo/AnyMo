_base_=['./default_runtime.py']
nb_code = 2048
unit_length=4
audio_unit_length=8
model_size = '3B' 
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
}

# model
model=dict(
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
        down_t=3, norm=None, stride_t=2, width=512, 
        activation='relu', depth=3, dilation_growth_rate=3,
        output_emb_width=llama_configs[model_size]['n_embd'],
    ),
    max_motion_len=300, 
    motion_downsample_ratio=unit_length, 
    audio_downsample_ratio=audio_unit_length, 
    stage=4, 
    # cond_types=['audio', 'text']
    # cond_types=['audio', 'text', 'traj']
    # cond_types=['audio', 'traj']
    cond_types=['audio']
)

# =============================================
# dataset
# =============================================

test_dataloader=dict(
    batch_size=2, 
    num_workers=4, 
    pin_memory=True,
    # batch_size=1, 
    # num_workers=0, 
    persistent_workers=True, 
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='anymo_collate_fn_v1'), 

    dataset=dict(
        type='AnyMoDataset_v1',
        data_path_file='configs/data_paths/speak_test.txt', 
        unit_length=unit_length, 
        mean_path=mean_path,
        std_path=std_path,
        traj_mean_path=traj_mean_path, 
        traj_std_path=traj_std_path, 
        audio_unit_length=audio_unit_length, 
        min_motion_length=60, 
        max_motion_length=300,
        # interval=10, 
    ), 
)

test_evaluator=dict(
    type='Audio2MotionMetric', 
    tmr_cfg_path='work_dirs/temos_50e/temos.py', 
    tmr_ckpt_path='work_dirs/temos_50e/epoch_50.pth'
)
val_cfg = None
default_hooks=dict(
    logger=dict(type='LoggerHook', interval=5),
)
