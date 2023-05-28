
import mmcv
from mmcv import Config
from mmpose.datasets.dataset_info import DatasetInfo
import numpy as np
from mmpose.core import imshow_bboxes, imshow_keypoints
import cv2

import os
from datetime import datetime

if __name__=="__main__":
    preds = np.load("preds.npy")
    targets = np.load("targets.npy")
    kpts = np.load("kpts.npy", allow_pickle=True)
    distances = np.load("distances.npy")
    cfg = Config.fromfile('configs/meter/hrnetv2_w18_meter_256x256.py')
    dataset_info = DatasetInfo(cfg.dataset_info)
    skeleton = dataset_info.skeleton
    pose_kpt_color = dataset_info.pose_kpt_color
    pose_link_color = dataset_info.pose_link_color
    pose_kpt_color_gt = np.array([[240, 230, 255], [240, 230, 255], [240, 230, 255], [240, 230, 255]])
    pose_link_color_gt = pose_kpt_color_gt

    now = datetime.now()
    #now = now.strftime("%d-%m-%y_%H-%M")
    now = now.strftime("%d-%m-%y_%H")
    imdir = "./vis_results/meter_"+now

    if not os.path.exists(imdir):
        os.makedirs(imdir)

    for pred, target, distance, kpt in zip(preds, targets, distances, kpts):
        impath = kpt["image_paths"]
        imname =  impath.split("/")[-1]
        impath_new = os.path.join(imdir, imname)

        im = mmcv.imread(impath)
        im = imshow_keypoints(img=im, pose_result=[kpt["keypoints"]], kpt_score_thr=0.3,
            skeleton=skeleton, pose_kpt_color=pose_kpt_color, pose_link_color=
            pose_link_color
        )

        prob_of_target_points = np.ones((target.shape[0], 1)) 
        target = np.concatenate((target, prob_of_target_points), axis= -1)

        im = imshow_keypoints(img=im, pose_result=[target], kpt_score_thr=0.01,
            skeleton=skeleton, pose_kpt_color=pose_kpt_color_gt, pose_link_color= pose_link_color_gt)

        boxes = np.array([kpt["center"]+ kpt["scale"]])
        #boxes = [kpt["center"]+ kpt["scale"]]
        #imshow_bboxes(img=im, bboxes=boxes,labels="{:.2f}".format(distance), out_file="test2.png")
        font = cv2.FONT_HERSHEY_SIMPLEX
        #imshape = (im.shape[0] - 20, im.shape[1] - 20)
        imshape = (100,500)
        cv2.putText(im, "PCK: {:.4f}".format(distance) ,imshape , font, 0.8,(0 ,255, 0),2,cv2.LINE_AA)

        mmcv.imwrite(im, impath_new)
    print(type(preds))
    
    
    

