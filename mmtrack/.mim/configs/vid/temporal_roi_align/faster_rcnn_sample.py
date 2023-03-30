_base_ = [
    '../../_base_/models/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/imagenet_vid_fgfa_style.py',
    '../../_base_/default_runtime.py'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data_ann_root = '/ds-av/public_datasets/imagenet/pre/ILSVRC2015/COCO-Annotations/'
data_root = "/ds-av/public_datasets/imagenet/raw/Data/"
model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='TemporalRoIAlign',
                num_most_similar_points=2,
                num_temporal_attention_blocks=4,
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=3,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)))))

# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=[
        dict(
            type='ImagenetVIDDataset',
            ann_file=data_ann_root+'imagenet_vid_train.json',
            img_prefix=data_root+'VID/',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=9,
                filter_key_img=True,
                method='bilateral_uniform'),
            pipeline=train_pipeline
            ),
        dict(
            type='ImagenetVIDDataset',
            load_as_video=False,
            ann_file=data_ann_root+'imagenet_det_30plus1cls.json',
            img_prefix=data_root+'DET/',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=train_pipeline
            )
    ],
    val=dict(
        type='ImagenetVIDDataset',
        ann_file=data_ann_root+'imagenet_vid_val.json',
        img_prefix=data_root+'VID/',
        ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'),
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type='ImagenetVIDDataset',
        ann_file=data_ann_root+'imagenet_vid_val.json',
        img_prefix=data_root+'VID/',
        ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'),
        pipeline=test_pipeline,
        test_mode=True))

# optimizer
optimizer = dict(type='SGD', lr=0.01/4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5])
# runtime settings
total_epochs = 7
evaluation = dict(metric=['bbox'], interval=7)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='vid_det',
                name='fcnn_simple'))
    ]
)