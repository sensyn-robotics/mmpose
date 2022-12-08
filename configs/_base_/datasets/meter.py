dataset_info = dict(
    dataset_name='meter',
    paper_info=dict(
        author='Ben Howells, James Charles and Roberto Cipolla',
        title='Annotated Synthetic Gauges ',
        container='2011 IEEE international conference on computer '
        'vision workshops (ICCV workshops)',
        year='2011',
        homepage='http://jjcvision.com/projects/gauge_reading.html'
    ),
    keypoint_info={
        0:
        dict(name='min', id=0, color=[51, 153, 255], type='', swap='max'),
        1:
        dict(name='max', id=1, color=[0, 255, 0], type='', swap='min'),
        2:
        dict(name='center', id=2, color=[255, 128, 0], type='', swap=''),
        3:
        dict(name='tip', id=3, color=[255, 0, 255], type='', swap=''),
    },
    skeleton_info={
        0:
        dict(link=('min', 'center'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('min', 'max'), id=1, color=[0, 0, 255]),
        2:
        dict(link=('max', 'center'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('center', 'tip'), id=3, color=[255, 0, 128]),
    },
    joint_weights=[1.] * 4,
    sigmas=[])
