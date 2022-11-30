#from mmcv import Config
#'./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
#cfg = Config.fromfile(
#    "/home/ahmed/work/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py"
#)
_base_="/home/ahmed/work/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py"
#_base_="./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py"
data_root = '/home/ahmed/work/mmpose/data/coco_tiny'

# set basic configs

data_root = '/home/ahmed/work/mmpose/data/coco_tiny'
work_dir = '/home/ahmed/work/mmpose/work_dirs/hrnet_w32_coco_tiny_256x192'
gpu_ids = range(1)
seed = 0

# set log interval
#log_config.interval = 1

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
total_epochs = 40

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
        type='TopDownCOCTinyDataset',
        ann_file=f'{data_root}/val.json',
        img_prefix=f'{data_root}/images/'),
        #data_cfg=data_cfg,
        #pipeline=val_pipeline,
        #dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/images/'),
        #data_cfg=data_cfg,
        #pipeline=test_pipeline,
        #dataset_info={{_base_.dataset_info}}),
)


#data.samples_per_gpu = 16
#data.val_dataloader = dict(samples_per_gpu=16)
#data.test_dataloader = dict(samples_per_gpu=16)
#
## set dataset configs
#data.train.type = 'TopDownCOCOTinyDataset'
#data.train.ann_file = f'{cfg.data_root}/train.json'
#data.train.img_prefix = f'{cfg.data_root}/images/'
#
#data.val.type = 'TopDownCOCOTinyDataset'
#data.val.ann_file = f'{cfg.data_root}/val.json'
#data.val.img_prefix = f'{cfg.data_root}/images/'
#
#data.test.type = 'TopDownCOCOTinyDataset'
#data.test.ann_file = f'{cfg.data_root}/val.json'
#data.test.img_prefix = f'{cfg.data_root}/images/'

#print(cfg.pretty_text)

