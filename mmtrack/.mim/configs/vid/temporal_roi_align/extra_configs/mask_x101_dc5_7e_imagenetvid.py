_base_ = ['./raw_data_mask_r50_imagenetvid.py']
model = dict(
    detector=dict(
        backbone=dict(
            type='ResNeXt',
            depth=101,
            groups=64,
            base_width=4,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://resnext101_64x4d'))))
num_gpus=8/4
optimizer = dict(type='SGD', lr=0.01/num_gpus, momentum=0.9, weight_decay=0.0001)