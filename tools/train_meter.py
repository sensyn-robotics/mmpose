
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)

from mmdet.apis import inference_detector, init_detector
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.apis import train_model
import mmcv
from mmcv import Config
from mmpose.datasets.builder import DATASETS
import sys
import pprint
import os
#cfg = Config.fromfile('./configs/cocotiny/hrnet_w32_cocotiny_256x192.py')

root_dir = "/home/ahmed/work/mmpose/"
root_dir2 = "/home/ahmed/work/mmdetection/"
cfg = Config.fromfile('./configs/meter/hrnetv2_w18_meter_256x256.py')
print(cfg.pretty_text)

det_config = os.path.join(root_dir2, 'configs/yolox/yolox_tiny_8x8_300e_coco.py')

det_checkpoint = '/home/ahmed/work/mmdetection/checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
# build dataset
print(DATASETS)
#sys.exit()

#pose_model = init_pose_model(cfg, pose_checkpoint)
#det_model = init_detector(det_config, det_checkpoint)

datasets = [build_dataset(cfg.data.train)]

# build model
model = build_posenet(cfg.model)

# create work_dir
mmcv.mkdir_or_exist(cfg.work_dir)

# train model
train_model(
    model, datasets, cfg, distributed=False, validate=False, meta=dict())
