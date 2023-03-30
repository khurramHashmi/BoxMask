# Copyright (c) OpenMMLab. All rights reserved.

'''
    Several changes done in the script to visualize specific directory predictions
    Use Train frames in test to visaulize training process
    Currently, everything is commented to use orignal script
    MODIFIED FUNCTION IS PLACED AT THE END AND IS COMMENTED AT THE MOMENT
'''
from mmdet.datasets import DATASETS
from mmdet.datasets.api_wrappers import COCO

from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID


@DATASETS.register_module()
class ImagenetVIDDataset(CocoVideoDataset):
    """ImageNet VID dataset for video object detection."""

    CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
               'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
               'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
               'zebra')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotations from COCO/COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO/COCOVID api.
        """

        if self.load_as_video:
            data_infos = self.load_video_anns(ann_file)
            print(f'loaded_VIDEO_anns: {len(data_infos)}')

        else:
            data_infos = self.load_image_anns(ann_file)
            print(f':loaded_IMAGE_anns {len(data_infos)}')

        return data_infos

    def load_image_anns(self, ann_file):
        """Load annotations from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO api.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        all_img_ids = self.coco.get_img_ids()
        self.img_ids = []
        data_infos = []
        for img_id in all_img_ids:
            info = self.coco.load_imgs([img_id])[0]
            info['filename'] = info['file_name']
            if info['is_vid_train_frame']:
                self.img_ids.append(img_id)
                data_infos.append(info)

        return data_infos

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                if self.test_mode:
                    assert not info['is_vid_train_frame'], \
                        'is_vid_train_frame must be False in testing'
                    self.img_ids.append(img_id)
                    data_infos.append(info)
                elif info['is_vid_train_frame']:
                    self.img_ids.append(img_id)
                    data_infos.append(info)
        return data_infos

    '''
        FUNCTION MODIFIED TO VISUALIZE DIFFERENT SPECIFIC PREDICTIONS AND LOADING SPECIFIC VIDEOS
    '''
    # def load_video_anns(self, ann_file):
    #     """Load annotations from COCOVID style annotation file.
    #
    #     Args:
    #         ann_file (str): Path of annotation file.
    #
    #     Returns:
    #         list[dict]: Annotation information from COCOVID api.
    #     """
    #     self.coco = CocoVID(ann_file)
    #     self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
    #     self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
    #
    #     data_infos = []
    #     self.vid_ids = self.coco.get_vid_ids()
    #     self.img_ids = []
    #
    #     # CHANGE CODE FOR MODIFYING ANNOTATIONS LENGTH STARTS HERE
    #     # print(f"Length of Vidoes : {len(self.vid_ids)}")
    #     # self.vid_ids = [self.vid_ids[0]]
    #     # print(f"Transformed of Vidoes : {(self.vid_ids)}")
    #     # CHANGE CODE FOR MODIFYING ANNOTATIONS LENGTH ENDS HERE
    #
    #     for vid_id in self.vid_ids:
    #         img_ids = self.coco.get_img_ids_from_vid(vid_id)
    #
    #         for img_id in img_ids:
    #             info = self.coco.load_imgs([img_id])[0]
    #             info['filename'] = info['file_name']
    #             if self.test_mode:
    #                 # TEMP CHANGE HERE TO ASSERT TRAIN FILE IN TEST
    #                 assert not info['is_vid_train_frame'], \
    #                     'is_vid_train_frame must be False in testing'
    #                 # Code change here
    #                 # if "ILSVRC2015_val_00007034" not in info['filename']:
    #                 #     # print(f"Filename : {info['filename']}")
    #                 #     continue
    #                 # # else:
    #                 # #Code change here
    #                 self.img_ids.append(img_id)
    #                 data_infos.append(info)
    #             # elif info['is_vid_train_frame']:
    #             #     self.img_ids.append(img_id)
    #             #     data_infos.append(info)
    #
    #             # CHANGE JUST TO VISUALIZE GT OF TEST IMAGES UPPER ELIF IS ORIGNAL
    #             else:
    #                 # if "ILSVRC2015_val_00160000" not in info['filename']:
    #                 #     # print(f"Filename : {info['filename']}")
    #                 #     continue
    #                 self.img_ids.append(img_id)
    #                 data_infos.append(info)
    #             # CHANGE JUST TO VISUALIZE GT OF TEST IMAGES UPPER ELIF IS ORIGNAL ENDS HERE
    #
    #     # import sys
    #     # sys.exit()
    #     return data_infos

    '''
        FUNCTION MODIFIED TO VISUALIZE DIFFERENT SPECIFIC PREDICTIONS AND LOADING SPECIFIC VIDEOS ENDS HERE
    '''