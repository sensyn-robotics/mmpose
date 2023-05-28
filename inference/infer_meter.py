
import mmcv
from mmcv import Config
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)

from mmdet.apis import inference_detector, init_detector

import numpy as np
import os
import cv2
import glob
from pathlib import Path
from datetime import date


def get_line_vector_from_point(fromPoint, toPoint):
    line_vector = toPoint - fromPoint
    return line_vector / np.linalg.norm(line_vector)

def get_angles(keypoints):
    min, max, center, tip = keypoints[0][:2], keypoints[1][:2], keypoints[2][:2], keypoints[3][:2]
    linevector_center_to_min = get_line_vector_from_point(center, min)
    linevector_center_to_max = get_line_vector_from_point(center, max)
    linevector_center_to_tip = get_line_vector_from_point(center, tip)

    get_angle_betn_lines = lambda x, y: np.arccos(np.dot(x, y)) / np.pi * 180
    angle_test = get_angle_betn_lines(np.array([1,0]), np.array([0,1]))
    assert int(angle_test) == 90
    angle_min_center = get_angle_betn_lines(linevector_center_to_min, linevector_center_to_tip)
    angle_min_max = get_angle_betn_lines(linevector_center_to_min, linevector_center_to_max)

    return angle_min_center, angle_min_max

def write_angle_on_bbox(img, bbox, label):
    #bbox = pose_results[0]["bbox"]
    bbox_int = bbox[:4].astype(np.int32)
    font_scale = 0.6
    thickness = 1
    text_color = "green"

    # roughly estimate the proper font size
    text_size, text_baseline = cv2.getTextSize(label,
                                               cv2.FONT_HERSHEY_DUPLEX,
                                               font_scale, thickness)
    text_x1 = bbox_int[0]
    text_y1 = max(0, bbox_int[1] - text_size[1] - text_baseline)
    text_x2 = bbox_int[0] + text_size[0]
    text_y2 = text_y1 + text_size[1] + text_baseline
    cv2.putText(img, label, (text_x1, text_y2 - text_baseline),
                cv2.FONT_HERSHEY_DUPLEX, font_scale,
                mmcv.color_val(text_color), thickness)
    return img


if __name__=="__main__":
    root_dir = "."
    root_dir2 = "../mmdetection"
    cfg = Config.fromfile(os.path.join(root_dir, 'configs/meter/hrnetv2_w18_meter_256x256.py'))
    pose_checkpoint = os.path.join(root_dir, 'work_dirs/hrnet_w32_meter_256x192/latest.pth')
    det_config = os.path.join(root_dir2, 'configs/yolox/yolox_tiny_8x8_300e_coco.py')
    det_checkpoint = '../mmdetection/checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

    pose_model = init_pose_model(cfg, pose_checkpoint)
    det_model = init_detector(det_config, det_checkpoint)

    #img = os.path.join(root_dir, 'tests/data/meter/scale_9_meas_0.png')
    #imdir = "/home/ahmed/work/coco-annotator/datasets/meter/test"

    imdir = "/home/ahmed/Downloads/meters_new"
    #imdir = "clahes"

    meterTestDir = os.path.join(root_dir, "tests/data/meter")
    for img in glob.glob(os.path.join(imdir,"*.JPG")):
        #img = "clahes/clahe_0.jpg"
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
        vis_result_im = vis_pose_result(
            pose_model,
            img,
            pose_results,
            kpt_score_thr=0.1,
            dataset='MeterDataset',
            show=False
        )
        if not len(pose_results)==0:
            for i in range(len(pose_results)):
                bbox = pose_results[i]["bbox"]
                keypoints = pose_results[i]["keypoints"]
                angle_min_center, angle_min_max  = get_angles(keypoints)
                print(f"angle_min_max {angle_min_max}")
                degree_sign = u'\N{DEGREE SIGN}'
                vis_result_im = write_angle_on_bbox(img=vis_result_im, bbox=bbox, label="Angle: {:.2f} degree".format(angle_min_center))
        imName = img.split('/')[-1]

        date_today = date.today().strftime("%d-%m-%y")
        print(date_today)
        dirname = 'vis_results/meters_'+date_today
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = os.path.join(root_dir, dirname, imName)
        print(filename)
        print(pose_results)
        cv2.imwrite(filename, vis_result_im)


