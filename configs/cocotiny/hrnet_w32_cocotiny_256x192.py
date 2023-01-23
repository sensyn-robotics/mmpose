_base_="../body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py"
data_root = '../../data/coco_tiny'

#custom_imports = dict(
#    imports=['.mmpose.datasets.datasets.top_down.topdown_coco_tiny_dataset.py'],
#    allow_failed_imports=False)


# set basic configs

data_root = '../../data/coco_tiny'
work_dir = '../../work_dirs/hrnet_w32_coco_tiny_256x192'
gpu_ids = range(1)
seed = 0

# set log interval
#log_config.interval = 1
log_config = dict(
    interval=1
)

# set evaluation configs
evaluation = dict(interval=10, metric='PCK', save_best='PCK')
#evaluation.interval = 10
#evaluation.metric = 'PCK'
#evaluation.save_best = 'PCK'

# set learning rate policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[17, 35])
total_epochs = 2

# set batch size
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    train=dict(
        type='TopDownCOCOTinyDataset',
        ann_file=f'{data_root}/train.json',
        img_prefix=f'{data_root}/images/'),
        #data_cfg=data_cfg,
        #pipeline=train_pipeline,
        #dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCOCOTinyDataset',
        ann_file=f'{data_root}/val.json',
        img_prefix=f'{data_root}/images/'),
        #data_cfg=data_cfg,
        #pipeline=val_pipeline,
        #dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCOCODataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/images/'),
        #data_cfg=data_cfg,
        #pipeline=test_pipeline,
        #dataset_info={{_base_.dataset_info}}),
)

