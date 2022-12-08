# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    # Basic configurations of the executor
    name='Meter Reading',
    camera_id=0,
    synchronous=False,
    # Define nodes.
    # The configuration of a node usually includes:
    #   1. 'type': Node class name
    #   2. 'name': Node name
    #   3. I/O buffers (e.g. 'input_buffer', 'output_buffer'): specify the
    #       input and output buffer names. This may depend on the node class.
    #   4. 'enable_key': assign a hot-key to toggle enable/disable this node.
    #       This may depend on the node class.
    #   5. Other class-specific arguments
    nodes=[
        # 'PoseTrackerNode':
        # This node performs object detection and pose tracking. Object
        # detection is performed every several frames. Pose estimation
        # is performed for every frame to get the keypoint as well as the
        # interval bbox when object detection is not performed.
        dict(
            type='PoseTrackerNode',
            name='pose tracker',
            det_model_config="/home/ahmed/work/mmdetection/configs/yolox/yolox_tiny_8x8_300e_coco.py",
            det_model_checkpoint='/home/ahmed/work/mmdetection/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
            pose_model_config='/home/ahmed/work/mmpose/configs/meter/hrnetv2_w18_meter_256x256.py',
            pose_model_checkpoint='/home/ahmed/work/mmpose/work_dirs/hrnet_w32_meter-test_256x192/latest.pth',
            det_interval=10,
            class_ids=[75],
            #labels=['person'],
            smooth=True,
            device='cuda:0',
            input_buffer='_input_',  # `_input_` is an executor-reserved buffer
            output_buffer='human_pose'),
        # 'ObjectAssignerNode':
        # This node binds the latest model inference result with the current
        # frame. (This means the frame image and inference result may be
        # asynchronous).
        dict(
            type='ObjectAssignerNode',
            name='object assigner',
            frame_buffer='_frame_',  # `_frame_` is an executor-reserved buffer
            object_buffer='human_pose',
            output_buffer='frame'),
        # 'ObjectVisualizerNode':
        # This node draw the pose visualization result in the frame image.
        # Pose results is needed.
        dict(
            type='ObjectVisualizerNode',
            name='object visualizer',
            enable_key='v',
            input_buffer='frame',
            output_buffer='vis'),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='instruction',
            enable_key='h',
            enable=True,
            input_buffer='vis',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:',
                '"v": Pose estimation result visualization',
                '"h": Show help information',
                '"m": Show diagnostic information', '"q": Exit'
            ],
        ),
        # 'MonitorNode':
        # This node show diagnostic information in the frame image. It can
        # be used for debugging or monitoring system resource status.
        dict(
            type='MonitorNode',
            name='monitor',
            enable_key='m',
            enable=False,
            input_buffer='vis_notice',
            output_buffer='display'),
        # 'RecorderNode':
        # This node save the output video into a file.
        dict(
            type='RecorderNode',
            name='recorder',
            out_video_file='webcam_meter.mp4',
            input_buffer='display',
            output_buffer='_display_'
            # `_display_` is an executor-reserved buffer
        )
    ])
