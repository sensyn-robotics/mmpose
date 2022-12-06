_base_ = '/home/ahmed/work/mmpose/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_meter_256x256.py'

work_dir="/home/ahmed/work/mmpose/work_dirs/hrnet_w32_meter-test_256x192"
gpu_ids = range(1)
seed = 0

total_epochs = 100
log_config = dict(
    interval=1,
)

channel_cfg = dict(
    num_output_channels=4,
    dataset_joints=4,
    dataset_channel=[
        list(range(4)),
    ],
    inference_channel=list(range(4)))

# model settings
#model = dict(
#    type='TopDown',
#    pretrained='open-mmlab://msra/hrnetv2_w18',
#    backbone=dict(
#        type='HRNet',
#        in_channels=3,
#        extra=dict(
#            stage1=dict(
#                num_modules=1,
#                num_branches=1,
#                block='BOTTLENECK',
#                num_blocks=(4, ),
#                num_channels=(64, )),
#            stage2=dict(
#                num_modules=1,
#                num_branches=2,
#                block='BASIC',
#                num_blocks=(4, 4),
#                num_channels=(18, 36)),
#            stage3=dict(
#                num_modules=4,
#                num_branches=3,
#                block='BASIC',
#                num_blocks=(4, 4, 4),
#                num_channels=(18, 36, 72)),
#            stage4=dict(
#                num_modules=3,
#                num_branches=4,
#                block='BASIC',
#                num_blocks=(4, 4, 4, 4),
#                num_channels=(18, 36, 72, 144),
#                multiscale_output=True),
#            upsample=dict(mode='bilinear', align_corners=False))),
#    keypoint_head=dict(
#        type='TopdownHeatmapSimpleHead',
#        in_channels=[18, 36, 72, 144],
#        in_index=(0, 1, 2, 3),
#        input_transform='resize_concat',
#        out_channels=channel_cfg['num_output_channels'],
#        num_deconv_layers=0,
#        extra=dict(
#            final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1, )),
#        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#    train_cfg=dict(),
#    test_cfg=dict(
#        flip_test=True,
#        post_process='default',
#        shift_heatmap=True,
#        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    #dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=30,
        scale_factor=0.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

#val_pipeline = [
#    dict(type='LoadImageFromFile'),
#    dict(type='TopDownGetBboxCenterScale', padding=1.25),
#    dict(type='TopDownAffine'),
#    dict(type='ToTensor'),
#    dict(
#        type='NormalizeTensor',
#        mean=[0.485, 0.456, 0.406],
#        std=[0.229, 0.224, 0.225]),
#    dict(
#        type='Collect',
#        keys=['img'],
#        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
#]

#test_pipeline = val_pipeline
#dataset_type='CocoDataset'
dataset_type='MeterDataset'
data_root = '/home/ahmed/work/coco-annotator'
train_annotation_file_name="newMeterAnn2.json"
#train_annotation_file_name="newMeterAnnOneImageOnly.json"
#train_annotation_file_name="newMeterAnnTwoImageOnly.json"
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=f'{data_root}/datasets/{train_annotation_file_name}',
        img_prefix=f'{data_root}/datasets/meter-test',
        data_cfg=data_cfg,
    ),
    val=dict(
        type=dataset_type,
        ann_file=f'{data_root}/datasets/{train_annotation_file_name}',
        img_prefix=f'{data_root}/datasets/meter-test',
        data_cfg=data_cfg
    )
#        pipeline=val_pipeline,
#        dataset_info={{_base_.dataset_info}}),
#    test=dict(
#        type='FaceAFLWDataset',
#        ann_file=f'{data_root}/annotations/face_landmarks_aflw_test.json',
#        img_prefix=f'{data_root}/images/',
#        data_cfg=data_cfg,
#        pipeline=test_pipeline,
#        dataset_info={{_base_.dataset_info}}),
)
