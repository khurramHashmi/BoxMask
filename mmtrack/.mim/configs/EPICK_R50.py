checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'

model = dict(
    detector=dict(
        type='FasterRCNN',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
        neck=[
            dict(
                type='ChannelMapper',
                in_channels=[2048],
                out_channels=512,
                kernel_size=3),
            dict(
                type='DyHead',
                in_channels=512,
                out_channels=512,
                num_blocks=6,
                zero_init_offset=False)
        ],
        rpn_head=dict(
            type='RPNHead',
            in_channels=512,
            feat_channels=512,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[4, 8, 16, 32],
                ratios=[0.5, 1.0, 2.0],
                strides=[16]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=0.1111111111111111,
                loss_weight=1.0)),
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='TemporalRoIAlign',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16],
                num_most_similar_points=2,
                num_temporal_attention_blocks=4),
            bbox_head=dict(
                type='SelsaBBoxHead',
                in_channels=512,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=351,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                num_shared_fcs=3,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16))),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=6000,
                max_per_img=600,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=6000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0001,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))),
    type='SELSA')

dataset_type = 'CocoVideoDataset'
classes=(
            'pan', 'pan:dust', 'tap', 'plate', 'knife', 'bowl', 'spoon',
            'cupboard', 'drawer', 'fridge', 'lid', 'hand', 'onion',
            'onion:spring', 'pot', 'glass', 'water', 'fork', 'board:chopping',
            'bag', 'sponge', 'spatula', 'cup', 'oil', 'bin', 'meat', 'potato',
            'bottle', 'container', 'tomato', 'salt', 'cloth', 'sink',
            'door:kitchen', 'pasta', 'dish:soap', 'food', 'kettle', 'box',
            'carrot', 'sauce', 'colander', 'milk', 'rice', 'garlic', 'pepper',
            'hob', 'dough', 'dishwasher', 'egg', 'cheese', 'bread', 'table',
            'salad', 'microwave', 'oven', 'cooker:slow', 'coffee', 'filter',
            'jar', 'rack:drying', 'chicken', 'tray', 'mixture', 'towel',
            'towel:kitchen', 'peach', 'skin', 'courgette', 'liquid:washing',
            'liquid', 'leaf', 'lettuce', 'leaf:mint', 'cutlery', 'scissors',
            'package', 'top', 'spice', 'tortilla', 'paper', 'machine:washing',
            'olive', 'sausage', 'glove:oven', 'peeler:potato', 'can', 'mat',
            'mat:sushi', 'vegetable', 'wrap:plastic', 'wrap', 'flour',
            'cucumber', 'curry', 'cereal', 'napkin', 'soap', 'squash', 'fish',
            'chilli', 'cover', 'sugar', 'aubergine', 'jug', 'heat', 'leek',
            'rubbish', 'ladle', 'mushroom', 'stock', 'freezer', 'light',
            'pizza', 'ball', 'yoghurt', 'chopstick', 'grape', 'ginger',
            'banana', 'oregano', 'tuna', 'kitchen', 'salmon', 'basket',
            'maker:coffee', 'roll', 'brush', 'lemon', 'clothes', 'grater',
            'strainer', 'bacon', 'avocado', 'blueberry', 'pesto', 'utensil',
            'bean:green', 'floor', 'lime', 'foil', 'grill', 'ingredient',
            'scale', 'paste:garlic', 'processor:food', 'nut:pine', 'butter',
            'butter:peanut', 'shelf', 'timer', 'rinse', 'tablecloth', 'switch',
            'powder:coconut', 'powder:washing', 'capsule', 'oat', 'tofu',
            'lighter', 'corn', 'vinegar', 'grinder', 'cap', 'support', 'cream',
            'content', 'tongs', 'pie', 'fan:extractor', 'raisin', 'toaster',
            'broccoli', 'pin:rolling', 'plug', 'button', 'tea', 'parsley',
            'flame', 'herb', 'base', 'holder:filter', 'thyme', 'honey',
            'celery', 'kiwi', 'tissue', 'time', 'clip', 'noodle', 'yeast',
            'hummus', 'coconut', 'cabbage', 'spinach', 'nutella', 'fruit',
            'dressing:salad', 'omelette', 'kale', 'paella', 'chip',
            'opener:bottle', 'shirt', 'chair', 'sandwich', 'burger:tuna',
            'pancake', 'leftover', 'risotto', 'pestle', 'sock', 'pea', 'apron',
            'juice', 'wine', 'dust', 'desk', 'mesh', 'oatmeal', 'artichoke',
            'remover:spot', 'coriander', 'mocha', 'quorn', 'soup', 'turmeric',
            'knob', 'seed', 'boxer', 'paprika', 'juicer:lime', 'guard:hand',
            'apple', 'tahini', 'finger', 'salami', 'mayonnaise', 'biscuit',
            'pear', 'mortar', 'berry', 'beef', 'squeezer:lime', 'tail',
            'stick:crab', 'supplement', 'phone', 'shell:egg', 'pith',
            'ring:onion', 'cherry', 'cake', 'sprout', 'almond', 'mint',
            'flake:chilli', 'cutter:pizza', 'nesquik', 'blender', 'scrap',
            'backpack', 'melon', 'breadcrumb', 'sticker', 'shrimp', 'smoothie',
            'grass:lemon', 'ketchup', 'slicer', 'stand', 'dumpling', 'watch',
            'beer', 'power', 'heater', 'basil', 'cinnamon', 'crisp',
            'asparagus', 'drink', 'fishcakes', 'mustard', 'caper', 'whetstone',
            'candle', 'control:remote', 'instruction', 'cork', 'tab', 'masher',
            'part', 'muffin', 'shaker:pepper', 'garni:bouquet', 'popcorn',
            'envelope', 'chocolate', 'spot', 'window', 'syrup', 'bar:cereal',
            'croissant', 'coke', 'stereo', 'alarm', 'recipe', 'handle',
            'sleeve', 'cumin', 'wire', 'label', 'fire', 'presser', 'air',
            'mouse', 'boiler', 'rest', 'tablet', 'poster', 'trousers', 'form',
            'rubber', 'rug', 'sheets', 'pepper:cayenne', 'waffle', 'pineapple',
            'turkey', 'alcohol', 'rosemary', 'lead', 'book', 'rim', 'gravy',
            'straw', 'hat', 'cd', 'slipper', 'casserole', 'ladder',
            'jambalaya', 'wall', 'tube', 'lamp', 'tarragon', 'heart', 'funnel',
            'whisk', 'driver:screw', 'trouser')

ann_root = "/netscratch/muralidhara/VID/data/"
data_root = "/netscratch/muralidhara/VID/data/"

train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
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
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
            type=dataset_type,
            ann_file=ann_root+"EPIC_Kitchen_train.json",
            img_prefix=data_root+"EPIC_Kitchen/",
            classes=classes,
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=9,
                filter_key_img=True,
                method='bilateral_uniform'),
            pipeline=train_pipeline),
    val=dict(
            type=dataset_type,
            ann_file=ann_root+"EPIC_Kitchen_valid.json",
            img_prefix=data_root+"EPIC_Kitchen/",
            classes=classes,
            ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'),
            pipeline=test_pipeline,
        test_mode=True),
    test=dict(
            type=dataset_type,
            ann_file=ann_root+"EPIC_Kitchen_valid.json",
            img_prefix=data_root+"EPIC_Kitchen/",
            classes=classes,
            ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'),
            pipeline=test_pipeline,
        test_mode=True))

optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.05,
            paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
    
#optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=1)

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
resume_from = None

workflow = [('train', 1)]

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[2, 5])
total_epochs = 7
evaluation = dict(metric=['bbox'], interval=7)
find_unused_parameters = True
gpu_ids = range(0, 1)