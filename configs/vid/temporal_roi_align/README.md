# BoxMask: Revisiting Bounding Box Supervision for Video Object Detection

## Introduction

[ALGORITHM]

```latex
@inproceedings{hashmi2023boxmask,
  title={BoxMask: Revisiting Bounding Box Supervision for Video Object Detection},
  author={Hashmi, Khurram Azeem and Pagani, Alain and Stricker, Didier and Afzal, Muhammad Zeshan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2030--2040},
  year={2023}
}
```

## Results and models on ImageNet VID dataset

We observed that the performance of this method has a fluctuation of about 0.5 mAP. The checkpoint provided below is the best one from two experiments.

Note that the numbers of selsa modules in this method and `SELSA` are 3 and 2 respectively. This is because another selsa modules improve this method by 0.2 points but degrade `SELSA` by 0.5 points. We choose the best settings for the two methods for a fair comparison.

|        Method        |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 |                                 Config                                  | Download |
|:--------------------:| :-----: | :-----: | :------: | :------------: |:---------:|:-----------------------------------------------------------------------:| :--------: |
|    TROI_R-50-DC5     |  pytorch  |   7e    | 4.14        | -            |   79.8    |     [config](selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py)     | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714-939fd657.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid_20210820_162714.log.json) |
| **BoxMask R-50-DC5** |  pytorch  |   7e    | 4.14        | -            |   **80.7**    | [config](boxMask_selsa_troialign_faster_rcnn_r50_dc5_7e_imagenetvid.py) | []() &#124; []() |
|      TROI_R-101-DC5       |  pytorch  |   7e    | 5.83        | -              |   82.6    |    [config](selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid.py)     | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid_20210822_111621-22cb96b9.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_r101_dc5_7e_imagenetvid_20210822_111621.log.json) |
|      TROI_X-101-DC5       |  pytorch  |   7e    | 9.74        | -              |   84.1    |    [config](selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid.py)     | [model](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid_20210822_164036-4471ac42.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vid/temporal_roi_align/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid_20210822_164036.log.json) |
