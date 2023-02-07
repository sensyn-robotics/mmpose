# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from mmcv import Config, deprecated_api_warning

from mmpose.datasets.builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class MeterDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Face AFLW dataset for top-down face keypoint localization.

    "Annotated Facial Landmarks in the Wild: A Large-scale,
    Real-world Database for Facial Landmark Localization".
    In Proc. First IEEE International Workshop on Benchmarking
    Facial Image Analysis Technologies, 2011.

    The dataset loads raw images and apply specified transforms
    to return a dict containing the image tensors and other information.

    The landmark annotations follow the 19 points mark-up. The definition
    can be found in `https://www.tugraz.at/institute/icg/research`
    `/team-bischof/lrs/downloads/aflw/`

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/meter.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)
        self.dataset_name = 'meter-test'
        self.ann_info['use_different_joint_weights'] = False
        self.db = self._get_db()
        print(f'=> ann_file: {self.ann_file}')
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        for i, img_id in enumerate(self.img_ids):

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                image_file = osp.join(self.img_prefix, self.id2name[img_id])
                print(f"{image_file} analysis start")
                #if self.test_mode:
                #    # 'box_size' is used as normalization factor
                #    assert 'box_size' in obj
                
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                # center = np.array(obj['center'])
                # scale = np.array([obj['scale'], obj['scale']])

                
                gt_db.append({
                    'image_file': image_file,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'box_size': obj['area'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])
        print(f"total bboxes {bbox_id}")
        print(f"get_db length {len(gt_db)}")
        return gt_db

    def _get_normalize_factor(self, box_sizes, *args, **kwargs):
        """Get normalize factor for evaluation.

        Args:
            box_sizes (np.ndarray[N, 1]): box size

        Returns:
            np.ndarray[N, 2]: normalized factor
        """

        return np.tile(box_sizes, [1, 2])

    def evaluate(self, results, res_folder=None, metric='PCK', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'NME']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")

        if res_folder is None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')


        kpts = []
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']
            batch_size = len(image_paths)
            for i in range(batch_size):
                kpts.append({
                    "keypoints":preds[i].tolist(),
                    "center":boxes[i][0:2].tolist(),
                    "scale":boxes[i][2:4].tolist(),
                    "area":float(boxes[i][4]),
                    "score":float(boxes[i][5]),
                    "bbox_id":bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value =OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value


