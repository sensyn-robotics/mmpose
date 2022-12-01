from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.apis import train_model
import mmcv
from mmcv import Config
from mmpose.datasets.builder import DATASETS
import sys
import pprint
pp = pprint.PrettyPrinter(width=41, compact=True)
#cfg = Config.fromfile('./configs/cocotiny/hrnet_w32_cocotiny_256x192.py')
cfg = Config.fromfile('./configs/cocotiny/hrnet_w32_cocotiny_256x192.py')
#print(cfg.pretty_text)
# build dataset
pp.pprint(DATASETS)
#sys.exit()
datasets = [build_dataset(cfg.data.train)]

# build model
model = build_posenet(cfg.model)

# create work_dir
mmcv.mkdir_or_exist(cfg.work_dir)

# train model
train_model(
    model, datasets, cfg, distributed=False, validate=True, meta=dict())
