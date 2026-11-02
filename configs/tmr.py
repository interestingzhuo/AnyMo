_base_=['./default_runtime.py']

randomness=dict(seed=123)
mean_path = 'data/mean_272.npy'
std_path = 'data/std_272.npy'

# model
model=dict(
    type='TMR', 
    temperature=0.1, 
    threshold_selfsim=0.9, 
    motion_encoder_cfg=dict(
        type='ACTORStyleEncoder',
        nfeats=272,
        latent_dim=256, 
        ff_size=1024, 
        num_layers=6,
        num_heads=4, 
        dropout=0.1, 
        activation='gelu'
    ), 
    motion_decoder_cfg=dict(
        type='ACTORStyleDecoder', 
        nfeats=272,
        latent_dim=256, 
        ff_size=1024, 
        num_layers=6,
        num_heads=4, 
        dropout=0.1, 
        activation='gelu'
    ), 
    text_encoder_cfg=dict(
        type='ACTORStyleEncoder',
        nfeats=768,
        latent_dim=256, 
        ff_size=1024, 
        num_layers=6,
        num_heads=4, 
        dropout=0.1, 
        activation='gelu'
    )
)

# =============================================
# dataset
# =============================================
unit_length=4
train_dataloader=dict(
    batch_size=4, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True, 
    sampler=dict(type='DefaultSampler', shuffle=True), 
    collate_fn=dict(type='anymo_text_collate_fn'), 
    dataset=dict(
        type='AnyMoTextDataset',
        data_path_file=[
            'configs/data_paths/omnihumo_motion_caption_dynamic.txt', 
            'configs/data_paths/omnihumo_motion_caption_static.txt'
        ], 
        unit_length=unit_length, 
        mean_path=mean_path,
        std_path=std_path,
        min_motion_length=60, 
        max_motion_length=300,
    ), 
)

val_dataloader=dict(
    batch_size=2, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True, 
    sampler=dict(type='DefaultSampler', shuffle=False), 
    collate_fn=dict(type='anymo_text_collate_fn'), 
    dataset=dict(
        type='AnyMoTextDataset',
        data_path_file='configs/data_paths/omnihumo_motion_caption_dynamic_test.txt', 
        unit_length=unit_length, 
        mean_path=mean_path,
        std_path=std_path,
        min_motion_length=60, 
        max_motion_length=300,
    ), 
)

test_dataloader=dict(
    batch_size=2, 
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True, 
    sampler=dict(type='DefaultSampler', shuffle=False), 
    collate_fn=dict(type='anymo_text_collate_fn'), 
    dataset=dict(
        type='AnyMoTextDataset',
        data_path_file='configs/data_paths/omnihumo_motion_caption_dynamic_test.txt', 
        unit_length=unit_length, 
        mean_path=mean_path,
        std_path=std_path,
        min_motion_length=60, 
        max_motion_length=300,
    ), 
)

val_evaluator=dict(
    type='T2MRetrievalMetric', 
    threshold_selfsim_metrics=0.95, 
)
test_evaluator=val_evaluator

# optimizer
optim_wrapper=dict(
    type='AmpOptimWrapper',
    dtype='float32',
    # dtype='bfloat16',
    optimizer=dict(
        type='AdamW', 
        lr=1e-4, 
        betas=(0.9, 0.99), weight_decay=0.0))
# training schedule
param_scheduler=[
    dict(type='LinearLR', by_epoch=False, start_factor=1e-5, begin=0, end=1000), 
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=50)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1e8)
default_hooks=dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, max_keep_ckpts=3, interval=10), 
    logger=dict(type='LoggerHook', interval=50),
)
