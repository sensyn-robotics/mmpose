from mmcv import Config
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)

from mmdet.apis import inference_detector, init_detector
import os
import cv2
import glob


root_dir = "/home/ahmed/work/mmpose/"
root_dir2 = "/home/ahmed/work/mmdetection/"
cfg = Config.fromfile(os.path.join(root_dir, 'configs/meter/hrnetv2_w18_meter_256x256.py'))
pose_checkpoint = os.path.join(root_dir, 'work_dirs/hrnet_w32_meter_256x192/latest.pth')
det_config = os.path.join(root_dir2, 'configs/yolox/yolox_tiny_8x8_300e_coco.py')
#det_config = os.path.join(root_dir, 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py')
det_checkpoint = '/home/ahmed/work/mmdetection/checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
#det_checkpoint = '/home/ahmed/work/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

pose_model = init_pose_model(cfg, pose_checkpoint)
det_model = init_detector(det_config, det_checkpoint)

img = os.path.join(root_dir, 'tests/data/meter/scale_9_meas_0.png')
imdir = "/home/ahmed/work/coco-annotator/datasets/meter/test"
#imdir = os.path.join(root_dir, 'tests/data/meter/')

meterTestDir = os.path.join(root_dir, "tests/data/meter")
for img in glob.glob(os.path.join(imdir,"*.png")):
    img = "/home/ahmed/work/mmpose/temp/temp3.png"
    mmdet_results = inference_detector(det_model, img)
    person_results = process_mmdet_results(mmdet_results, cat_id=75)
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        bbox_thr=0.3,
        format='xyxy',
        dataset='MeterDataset'
    )
    vis_result = vis_pose_result(
        pose_model,
        img,
        pose_results,
        kpt_score_thr=0.,
        dataset='MeterDataset',
        show=False
    )
    #vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
    imName = img.split('/')[-1]
    filename = os.path.join(root_dir,'vis_results/meter2', imName)
    print(pose_results)
    #break
    cv2.imwrite(filename, vis_result)
    break


