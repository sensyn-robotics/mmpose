from mmdet.apis import init_detector, inference_detector
import mmcv
import os.path as osp
#get mmdetection at one directory above the mmpose root directory
root_dir="../../mmdetection/"
# Specify the path to model config and checkpoint file
config_file = osp.join(root_dir,'configs/yolox/yolox_tiny_8x8_300e_coco.py')
checkpoint_file = osp.join(root_dir, 'checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth')

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'detection/meterImage1.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='../vis_results/result.jpg')

