_base_ = ['./mask_single_conv_MTROI.py']
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
