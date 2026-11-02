_base_=['./default_runtime.py']

code_dim = 512
nb_code = 2048
levels = [8, 8, 6, 5]

# nb_code = 1024
# levels = [8, 5, 5, 5]
find_unused_parameters=True
mean_path = 'data/mean_272.npy'
std_path = 'data/std_272.npy'

# model
model=dict(
    type='VQVAE', 
    encoder_cfg=dict(
        type='EncoderAttn', 
        input_emb_width=272, 
        output_emb_width=code_dim, 
        down_t=2, 
        stride_t=2, 
        width=code_dim, depth=3, 
        dilation_growth_rate=3, 
        activation='relu', 
        norm=None, 
    ), 
    decoder_cfg=dict(
        type='DecoderAttn', 
        input_emb_width=272, 
        output_emb_width=code_dim, 
        down_t=2, 
        width=code_dim, 
        depth=3, 
        dilation_growth_rate=3,
        activation='relu',
        norm=None, 
    ), 

    quantizer_cfg=dict(
        type='ResidualFSQ', levels=levels, dim=code_dim, num_quantizers=4, quantize_dropout_prob=0.2),

    loss_cfg=dict(type='ReconsLoss', lambda_recons=1.0, lambda_pos=0.5)
)

default_hooks=dict(
    checkpoint=dict(type='CheckpointHook', max_keep_ckpts=1, interval=50), 
    logger=dict(type='LoggerHook', interval=100),
)

unit_length = 4
train_dataloader=dict(
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='anymo_motion_collate_fn'), 
    dataset=dict(
        type='AnyMoMotionDataset',
        data_path_file=[
            'configs/data_paths/omnihumo_motion.txt', 
            'configs/data_paths/omnihumo_motion_caption_static.txt', 
            'configs/data_paths/omnihumo_motion_caption_speak.txt', 
            'configs/data_paths/omnihumo_motion_caption_dynamic.txt', 
            'configs/data_paths/omnihumo_motion_caption_dance.txt', 
            'configs/data_paths/mm_train.txt'
        ], 
        unit_length=unit_length, 
        window_size=-1,
        mean_path=mean_path,
        std_path=std_path,
        min_motion_length=60, 
        max_motion_length=300,
    ), 
)

val_dataloader=dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='motion_collate_fn'), 
    dataset=dict(
        type='MotionDataset',
        split_file='data/motionmillion/mini_test.pkl', 
        unit_length=unit_length, 
        window_size=-1,
        mean_path=mean_path,
        std_path=std_path,
        min_motion_length=60, 
        max_motion_length=300,
    ), 
)

test_dataloader=dict(
    
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),

    # batch_size=64,
    # collate_fn=dict(type='motion_collate_fn'), 
    # dataset=dict(
    #     type='MotionDataset',
    #     split_file='data/motionmillion/mini_test.pkl',
    #     unit_length=unit_length, 
    #     window_size=-1,
    #     mean_path=mean_path,
    #     std_path=std_path,
    #     repeat_time=1, 
    #     min_motion_length=60, 
    #     max_motion_length=300,
    # ), 

    batch_size=4,
    collate_fn=dict(type='anymo_motion_collate_fn'), 
    dataset=dict(
        type='AnyMoMotionDataset',
        data_path_file=[
            'configs/data_paths/omnihumo_vq_test.txt', 
            'configs/data_paths/omnihumo_motion_caption_dynamic_test.txt', 
            'configs/data_paths/omnihumo_motion_caption_static_test.txt', 
            'configs/data_paths/dance_test.txt', 
            'configs/data_paths/speak_test.txt'
        ], 
        unit_length=unit_length, 
        window_size=-1,
        mean_path=mean_path,
        std_path=std_path,
        min_motion_length=60, 
        max_motion_length=300,
    ), 
)

val_evaluator=dict(type='ReconsMetric')
test_evaluator=val_evaluator

# optimizer
optim_wrapper=dict(
    type='AmpOptimWrapper',
    dtype='float32',
    # dtype='bfloat16',
    optimizer=dict(
        type='AdamW', 
        lr=2e-4, 
        betas=(0.9, 0.99), weight_decay=0.0))

# training schedule
param_scheduler=[
    dict(type='LinearLR', by_epoch=False, start_factor=1e-5, begin=0, end=1000), 
    dict(type='MultiStepLR', by_epoch=True, milestones=[60, 140], gamma=0.3)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1e8)
