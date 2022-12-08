_base_ = '/home/ahmed/work/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_meter_256x192.py'

work_dir="/home/ahmed/work/mmpose/work_dirs/hrnet_w32_meter_256x192"
gpu_ids = range(1)
seed = 0

total_epochs = 100
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=True),
    ],
    init_kwargs={
        'entity': "ahmed",
        'project': "meterCheck"
        }

)

channel_cfg = dict(
    num_output_channels=4,
    dataset_joints=4,
    dataset_channel=[
        list(range(4)),
    ],
    inference_channel=list(range(4)))

evaluation = dict(interval=10, metric='PCK', save_best='PCK')
# set learning rate policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[17, 35])



dataset_type='MeterDataset'
#dataset_type='MeterDatasetCoco'
data_root = '/home/ahmed/work/coco-annotator'
train_annotation_file_name="meter_train.json"
test_annotation_file_name="meter_test.json"

samples_per_gpu = 16
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=samples_per_gpu),
    test_dataloader=dict(samples_per_gpu=samples_per_gpu),
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/datasets/{train_annotation_file_name}',
        img_prefix=f'{data_root}/datasets/meter/train',
        data_cfg={{_base_.data_cfg}},
        pipeline={{_base_.train_pipeline}},
        dataset_info={{_base_.dataset_info}}
        ),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/datasets/{train_annotation_file_name}',
        img_prefix=f'{data_root}/datasets/meter/train',
        data_cfg={{_base_.data_cfg}},
        pipeline={{_base_.val_pipeline}},
        dataset_info={{_base_.dataset_info}}
        ),

    test=dict(
        type='MeterDataset',
        ann_file=f'{data_root}/datasets/{test_annotation_file_name}',
        img_prefix=f'{data_root}/meter/test',
        data_cfg={{_base_.data_cfg}},
        pipeline={{_base_.test_pipeline}},
        dataset_info={{_base_.dataset_info}}
        )
)
