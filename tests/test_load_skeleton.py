#from mmpose.datasets.dataset_info import DatasetInfo
import mmcv
from mmpose.datasets import DatasetInfo
import numpy as np



if __name__=="__main__":

    config = mmcv.Config.fromfile("configs/meter/hrnetv2_w18_meter_256x256.py")
    print(type(config))

    print(config.data['test']['type'])
    
    dataset_info = config.data['test'].get('dataset_info', None)
    di = DatasetInfo(dataset_info)
    print(di.skeleton)
    print(dataset_info)
    print(dataset_info.keys())

