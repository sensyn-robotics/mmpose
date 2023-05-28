import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow
from mmpose.datasets.dataset_info import DatasetInfo
from mmcv import Config
from mmpose.core import imshow_bboxes, imshow_keypoints

import argparse
import errno
import os

from convertJsonToDataFrame import convertJsonToDf



def parse_args():
    parser = argparse.ArgumentParser(description='visualize results from result_json file')
    parser.add_argument('--result_json',
                        default=None,help='input json file'
                        )
    args = parser.parse_args()
    return args

#def get_skeleton(dataset_info):
def get_line_vector_from_point(fromPoint, toPoint):
    line_vector = toPoint - fromPoint
    return line_vector / np.linalg.norm(line_vector)


if __name__=="__main__":
    args = parse_args()
    filename = args.result_json
    if filename is None:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filename)

    cfg = Config.fromfile('configs/meter/hrnetv2_w18_meter_256x256.py')
    dataset_info = DatasetInfo(cfg.dataset_info)
    skeleton = dataset_info.skeleton 
    pose_kpt_color = dataset_info.pose_kpt_color
    pose_link_color = dataset_info.pose_link_color

    df = convertJsonToDf(filename=filename)
    print(df.columns)
    
    kpts = df["preds"][0]
    for kpts, box, impath in zip(df["preds"], df["boxes"], df["image_paths"]):
        print(impath)
        print(box) 
        kpts = np.array(kpts)
        min, max, center, tip = kpts[0][:2], kpts[1][:2], kpts[2][:2], kpts[3][:2]
        linevector_center_to_min = get_line_vector_from_point(center, min)
        linevector_center_to_max = get_line_vector_from_point(center, max)
        linevector_center_to_tip = get_line_vector_from_point(center, tip)

        get_angle_betn_lines = lambda x, y: np.arccos(np.dot(x, y)) / np.pi * 180
        angle_test = get_angle_betn_lines(np.array([1,0]), np.array([0,1]))
        angle_min_center = get_angle_betn_lines(linevector_center_to_min, linevector_center_to_tip)
        angle_min_max = get_angle_betn_lines(linevector_center_to_min, linevector_center_to_max)

        im = mmcv.imread(impath)
        im = imshow_keypoints(img=im, pose_result=[kpts], kpt_score_thr=0.3,
            skeleton=skeleton, pose_kpt_color=pose_kpt_color, pose_link_color=
            pose_link_color 
        )
        imshow_bboxes(img=im, bboxes=box,labels=str(angle_min_max), out_file="test2.png")
        mmcv.imwrite(im, "test.png")
        print(".")

    




