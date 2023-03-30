# BoxMask: Revisiting Bounding Box Supervision for Video Object Detection
[![License](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)

Official code for BoxMask: Revisiting Bounding Box Supervision for Video Object Detection, WACV 2023

[Paper](https://arxiv.org/pdf/2210.06008.pdf)

## Installation

Please refer to [install.md](docs/en/install.md) for install instructions.

## Getting Started

An exectuable MMTracking framework is required to run this code.
Please see [dataset.md](docs/en/dataset.md) and [quick_run.md](docs/en/quick_run.md) for the basic usage of MMTracking.

A Colab tutorial is provided. You may preview the notebook [here](./demo/MMTracking_Tutorial.ipynb) or directly run it on [Colab](https://colab.research.google.com/github/open-mmlab/mmtracking/blob/master/demo/MMTracking_Tutorial.ipynb).

There are also usage [tutorials](docs/en/tutorials/), such as [learning about configs](docs/en/tutorials/config.md), [an example about detailed description of vid config](docs/en/tutorials/config_vid.md), [an example about detailed description of mot config](docs/en/tutorials/config_mot.md), [an example about detailed description of sot config](docs/en/tutorials/config_sot.md), [customizing dataset](docs/en/tutorials/customize_dataset.md), [customizing data pipeline](docs/en/tutorials/customize_data_pipeline.md), [customizing vid model](docs/en/tutorials/customize_vid_model.md), [customizing mot model](docs/en/tutorials/customize_mot_model.md), [customizing sot model](docs/en/tutorials/customize_sot_model.md), [customizing runtime settings](docs/en/tutorials/customize_runtime.md) and [useful tools](docs/en/useful_tools_scripts.md).

### Video Object Detection

Supported Methods

- [x] [DFF](configs/vid/dff) (CVPR 2017)
- [x] [FGFA](configs/vid/fgfa) (ICCV 2017)
- [x] [SELSA](configs/vid/selsa) (ICCV 2019)
- [x] [Temporal RoI Align](configs/vid/temporal_roi_align) (AAAI 2021)
- [x] [BoxMask+Temporal RoI Align](configs/vid/temporal_roi_align/) (WACV 2023)

Supported Datasets

- [x] [ILSVRC](http://image-net.org/challenges/LSVRC/2017/)

## Acknowledgement
This codebase is heavily based on [MMtracking](https://github.com/open-mmlab/mmtracking) and [MMDetection](https://github.com/open-mmlab/mmdetection). We sincerely thank their efforts for providing such useful open source frameworks

## Citing BoxMask
Please cite our paper in your publications if it helps your research:
```
@inproceedings{hashmi2023boxmask,
  title={BoxMask: Revisiting Bounding Box Supervision for Video Object Detection},
  author={Hashmi, Khurram Azeem and Pagani, Alain and Stricker, Didier and Afzal, Muhammad Zeshan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2030--2040},
  year={2023}
}
```
