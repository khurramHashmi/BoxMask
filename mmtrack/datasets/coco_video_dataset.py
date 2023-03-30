'''
    Several changes done in the script to visualize specific directory predictions
    In parse_annotation folder, Add grabcut to visualize ground truth
    Also added code to draw bounding boxes and masks on Images
    All code commented at the moment
'''

# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import DATASETS, CocoDataset

from mmtrack.core import eval_mot
from mmtrack.utils import get_root_logger
from .parsers import CocoVID
import cv2
import time
from skimage.measure import approximate_polygon



@DATASETS.register_module()
class CocoVideoDataset(CocoDataset):
    """Base coco video dataset for VID, MOT and SOT tasks.

    Args:
        load_as_video (bool): If True, using COCOVID class to load dataset,
            otherwise, using COCO class. Default: True.
        key_img_sampler (dict): Configuration of sampling key images.
        ref_img_sampler (dict): Configuration of sampling ref images.
        test_load_ann (bool): If True, loading annotations during testing,
            otherwise, not loading. Default: False.
    """

    CLASSES = None


    def __init__(self,
                 load_as_video=True,
                 key_img_sampler=dict(interval=1),
                 ref_img_sampler=dict(
                     frame_range=10,
                     stride=1,
                     num_ref_imgs=1,
                     filter_key_img=True,
                     method='uniform',
                     return_key_img=True),
                 test_load_ann=False,
                 *args,
                 **kwargs):
        self.load_as_video = load_as_video
        self.key_img_sampler = key_img_sampler
        self.ref_img_sampler = ref_img_sampler
        self.test_load_ann = test_load_ann
        super().__init__(*args, **kwargs)
        self.logger = get_root_logger()

    def load_annotations(self, ann_file):
        """Load annotations from COCO/COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO/COCOVID api.
        """
        if not self.load_as_video:
            data_infos = super().load_annotations(ann_file)
        else:
            data_infos = self.load_video_anns(ann_file)
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
            if self.key_img_sampler is not None:
                img_ids = self.key_img_sampling(img_ids,
                                                **self.key_img_sampler)
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                #if img_id == 130968 or img_id == 130967:
                #      continue
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
        return data_infos

    def key_img_sampling(self, img_ids, interval=1):
        """Sampling key images."""
        return img_ids[::interval]

    def ref_img_sampling(self,
                         img_info,
                         frame_range,
                         stride=1,
                         num_ref_imgs=1,
                         filter_key_img=True,
                         method='uniform',
                         return_key_img=True):
        """Sampling reference frames in the same video for key frame.

        Args:
            img_info (dict): The information of key frame.
            frame_range (List(int) | int): The sampling range of reference
                frames in the same video for key frame.
            stride (int): The sampling frame stride when sampling reference
                images. Default: 1.
            num_ref_imgs (int): The number of sampled reference images.
                Default: 1.
            filter_key_img (bool): If False, the key image will be in the
                sampling reference candidates, otherwise, it is exclude.
                Default: True.
            method (str): The sampling method. Options are 'uniform',
                'bilateral_uniform', 'test_with_adaptive_stride',
                'test_with_fix_stride'. 'uniform' denotes reference images are
                randomly sampled from the nearby frames of key frame.
                'bilateral_uniform' denotes reference images are randomly
                sampled from the two sides of the nearby frames of key frame.
                'test_with_adaptive_stride' is only used in testing, and
                denotes the sampling frame stride is equal to (video length /
                the number of reference images). test_with_fix_stride is only
                used in testing with sampling frame stride equalling to
                `stride`. Default: 'uniform'.
            return_key_img (bool): If True, the information of key frame is
                returned, otherwise, not returned. Default: True.

        Returns:
            list(dict): `img_info` and the reference images information or
            only the reference images information.
        """
        assert isinstance(img_info, dict)
        if isinstance(frame_range, int):
            assert frame_range >= 0, 'frame_range can not be a negative value.'
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, list):
            assert len(frame_range) == 2, 'The length must be 2.'
            assert frame_range[0] <= 0 and frame_range[1] >= 0
            for i in frame_range:
                assert isinstance(i, int), 'Each element must be int.'
        else:
            raise TypeError('The type of frame_range must be int or list.')

        if 'test' in method and \
                (frame_range[1] - frame_range[0]) != num_ref_imgs:
            print_log(
                'Warning:'
                "frame_range[1] - frame_range[0] isn't equal to num_ref_imgs."
                'Set num_ref_imgs to frame_range[1] - frame_range[0].',
                logger=self.logger)
            self.ref_img_sampler[
                'num_ref_imgs'] = frame_range[1] - frame_range[0]

        if (not self.load_as_video) or img_info.get('frame_id', -1) < 0 \
                or (frame_range[0] == 0 and frame_range[1] == 0):
            ref_img_infos = []
            for i in range(num_ref_imgs):
                ref_img_infos.append(img_info.copy())
        else:
            vid_id, img_id, frame_id = img_info['video_id'], img_info[
                'id'], img_info['frame_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            left = max(0, frame_id + frame_range[0])
            right = min(frame_id + frame_range[1], len(img_ids) - 1)

            ref_img_ids = []
            if method == 'uniform':
                valid_ids = img_ids[left:right + 1]
                if filter_key_img and img_id in valid_ids:
                    valid_ids.remove(img_id)
                num_samples = min(num_ref_imgs, len(valid_ids))
                ref_img_ids.extend(random.sample(valid_ids, num_samples))
            elif method == 'bilateral_uniform':
                assert num_ref_imgs % 2 == 0, \
                    'only support load even number of ref_imgs.'
                for mode in ['left', 'right']:
                    if mode == 'left':
                        valid_ids = img_ids[left:frame_id + 1]
                    else:
                        valid_ids = img_ids[frame_id:right + 1]
                    if filter_key_img and img_id in valid_ids:
                        valid_ids.remove(img_id)
                    num_samples = min(num_ref_imgs // 2, len(valid_ids))
                    sampled_inds = random.sample(valid_ids, num_samples)
                    ref_img_ids.extend(sampled_inds)
            elif method == 'test_with_adaptive_stride':
                if frame_id == 0:
                    stride = float(len(img_ids) - 1) / (num_ref_imgs - 1)
                    for i in range(num_ref_imgs):
                        ref_id = round(i * stride)
                        ref_img_ids.append(img_ids[ref_id])
            elif method == 'test_with_fix_stride':
                if frame_id == 0:
                    for i in range(frame_range[0], 1):
                        ref_img_ids.append(img_ids[0])
                    for i in range(1, frame_range[1] + 1):
                        ref_id = min(round(i * stride), len(img_ids) - 1)
                        ref_img_ids.append(img_ids[ref_id])
                elif frame_id % stride == 0:
                    ref_id = min(
                        round(frame_id + frame_range[1] * stride),
                        len(img_ids) - 1)
                    ref_img_ids.append(img_ids[ref_id])
                img_info['num_left_ref_imgs'] = abs(frame_range[0]) \
                    if isinstance(frame_range, list) else frame_range
                img_info['frame_stride'] = stride
            else:
                raise NotImplementedError

            ref_img_infos = []
            for ref_img_id in ref_img_ids:
                ref_img_info = self.coco.load_imgs([ref_img_id])[0]
                ref_img_info['filename'] = ref_img_info['file_name']
                ref_img_infos.append(ref_img_info)
            ref_img_infos = sorted(ref_img_infos, key=lambda i: i['frame_id'])

        if return_key_img:
            return [img_info, *ref_img_infos]
        else:
            return ref_img_infos

    def get_ann_info(self, img_info):
        """Get COCO annotations by the information of image.

        Args:
            img_info (int): Information of image.

        Returns:
            dict: Annotation information of `img_info`.
        """

        
        
        """
            Only in case of Confusion matrix function
            else it this piece of code should be commented 
            The function sends data index from which image info needs to be retrieved
        """
        # img_info = self.data_infos[img_info]
        """
            confusion matrix logic ends here
        """

        img_id = img_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(img_info, ann_info)

    def prepare_results(self, img_info):
        """Prepare results for image (e.g. the annotation information, ...)."""
        results = dict(img_info=img_info)

        if not self.test_mode or self.test_load_ann:
            results['ann_info'] = self.get_ann_info(img_info)
        if self.proposals is not None:
            idx = self.img_ids.index(img_info['id'])
            results['proposals'] = self.proposals[idx]
        
        super().pre_pipeline(results)
        results['is_video_data'] = self.load_as_video
        return results

    def prepare_data(self, idx):
        """Get data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Data and annotations after pipeline with new keys introduced
            by pipeline.
        """
        img_info = self.data_infos[idx]
        
        if self.ref_img_sampler is not None:
            img_infos = self.ref_img_sampling(img_info, **self.ref_img_sampler)
            results = [
                self.prepare_results(img_info) for img_info in img_infos
            ]
        else:
            results = self.prepare_results(img_info)
        # print(f"VALUE results : {results}")
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotations after pipeline with new keys
            introduced by pipeline.
        """
        return self.prepare_data(idx)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
            pipeline.
        """
        return self.prepare_data(idx)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotations.

        Args:
            img_anfo (dict): Information of image.
            ann_info (list[dict]): Annotation information of image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
            labels, instance_ids, masks, seg_map. "masks" are raw
            annotations and not decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks = []
        gt_instance_ids = []
        rpn_masks = []

        #root paths needed for visualization
        # root_path_1 = "/ds-av/public_datasets/imagenet/raw/Data/VID/"
        # root_path_2 = "/ds-av/public_datasets/imagenet/raw/Data/DET/"

        #incase of Epic Kitchen Dataset
        # root_path_2 = "../dataset_annotations/EPIC_Kitchens/images/EPIC_Kitchen/"

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

                #forceful addition of mask , need to be modified later
                # if 'segmentation' in ann:
                #     gt_masks.append(ann['segmentation'])


                # gt_masks.append([[x1, y1, x1+w,y1, x1+w, y1+h, x1,y1+h, x1,y1]])


                #Code for another mask creation in which mask is smaller than the actual bounding box.
                #Scale factor 1 means mask is equal to the bounding box.
                #WHEN USING GRABCUT GT THIS PIECE OF CODE WILL BE COMMENTED
                scaling_factor_mask_rcnn = 1
                x_1 = int(x1 + w/scaling_factor_mask_rcnn)
                x2 = int(x1 + (w- w/scaling_factor_mask_rcnn))
                y_1 = int(y1 + h/scaling_factor_mask_rcnn)
                y2 = int(y1 + (h- h/scaling_factor_mask_rcnn))
                gt_masks.append([[x_1, y_1, x2, y_1, x2, y2, x_1, y2, x_1, y_1]])



                '''
                    ADDING GRABCUT TO GENERATE BBOXMASK GT CODE STARTS HERE
                '''
                # img=cv2.imread(root_path_2+img_info["file_name"])
                # if img is None:
                #     # print(f"USING PATH 2")
                #     img = cv2.imread(root_path_1 + img_info["file_name"])

                # # Now apply gabcut on image
                # mask_grabcut = np.zeros(img.shape[:2], dtype="uint8")
                # bgdModel = np.zeros((1, 65), np.float64)
                # fgdModel = np.zeros((1, 65), np.float64)
                # rect = (int(x1), int(y1), int(w), int(h))

                # cv2.grabCut(img, mask_grabcut, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                # mask_coordinates = np.where(mask_grabcut == cv2.GC_PR_FGD)
                # segmentation_coords = [None]*(len(mask_coordinates[0])+len(mask_coordinates[1]))
                # segmentation_coords[::2] = mask_coordinates[1]
                # segmentation_coords[1::2] = mask_coordinates[0]

                #Restricting the polygon coordinates to 20 at max to avoid memory load
                # point = np.zeros((len(mask_coordinates[1]),2))
                # point[:,0] = mask_coordinates[1]
                # point[:,1] = mask_coordinates[0]
                # tolerance = 0.0
                # if len(point) > 20:
                #     new_point = point
                #     while len(new_point) > 20 and tolerance < 100 :
                #         tolerance += 1.0
                #         new_point = approximate_polygon(np.asarray(point), tolerance=tolerance)
                #     segmentation_coords = [i for i in new_point]
                #
                # segmentation_coords = [x.tolist() for x in segmentation_coords]
                # segmentation_coords = sum(segmentation_coords, [])

                # if len(segmentation_coords) < 8:
                #     gt_masks.append([[x_1, y_1, x2, y_1, x2, y2, x_1, y2, x_1, y_1]])
                # else:
                #     gt_masks.append([segmentation_coords])
                # For visualization
                # print(f" Bbox Coord : {rect} and Segemntation Coord : {segmentation_coords}")
                '''
                    ADDING GRABCUT TO GENERATE BBOXMASK GT ENDS STARTS HERE
                '''
                # # Code for small mask ends here

                ## Code For creating mask for RPN with the default BBox information Starts here
                # scaling_factor_rpn= 4
                # x_1_r = int(x1 + w/scaling_factor_rpn)
                # x2_r = int(x1 + (w- w/scaling_factor_rpn))
                # y_1_r = int(y1 + h/scaling_factor_rpn)
                # y2_r = int(y1 + (h- h/scaling_factor_rpn))
                # rpn_masks.append([[x_1_r, y_1_r, x2_r, y_1_r, x2_r, y2_r, x_1_r, y2_r, x_1_r, y_1_r]])
                ## Code For creating mask for RPN with the default BBox information ENDS here


                # '''
                #     VISUALIZATION OF GROUND TRUTH CODE, COMMENT THIS IF NOT NEEDED
                # '''
                # #Verification of both paths VID and DET
                # img=cv2.imread(root_path_1+img_info["file_name"])
                # if img is None:
                #     print(f"USING PATH 2")
                #     img = cv2.imread(root_path_2 + img_info["file_name"])
                #
                # file_name = img_info["file_name"].split("/")[-1].split(".")[0]
                #
                # # print(f"Coords : {int(x1)}, {int(y1)}, {int(x1 + w)}, {int(y1 + h)}")
                # #Drawing bboxes on image
                # result = cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 5)
                # cv2.imwrite("browse_images/gt_bbox_" + file_name + ".png", result)
                # #
                #
                # '''
                #     VISUALIZING MASK CODE ON THE COMPLETE IMAGE STARTS HERE
                # '''
                # image = img.astype(np.uint32).copy()
                # image[int(y1):int(y1 + h),int(x1):int(x1 + w)] *= 3
                #
                # # for c in range(3):
                # #     image[:, :, c] = np.where(image[:, :, c] == 1,
                # #                               # image[:, :, c] * 0.8,
                # #                               image[:, :, c] * 1.8,
                # #                               # (1 - 0.5) + 0.5 * 0.1 * 255,
                # #                               image[:, :, c])
                # # masked = cv2.bitwise_and(img, img, mask=mask)
                # image = image.astype('float32')
                # # cv2.putText(image, CLASSES[index], (det_bboxes[0][0][0], det_bboxes[0][0][1] - 5),
                # #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # cv2.imwrite(f"browse_images/gt_mask_{file_name}.png", image)

                '''
                    VISUALIZING MASK CODE ON THE COMPLETE IMAGE ENDS HERE
                '''

                # # Now for mask
                # mask = np.zeros(img.shape[:2], dtype="uint8")
                # cv2.rectangle(mask, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), 255, -1)
                # # apply our mask -- notice how only the person in the image is
                # # cropped out.
                # print(f"BBOX : {gt_bboxes} and Mask Coords : {gt_masks}")
                # print(f"File name : {root_path_2 + img_info['file_name']}")
                #
                # masked = cv2.bitwise_and(img, img, mask=mask)
                # cv2.imwrite("browse_images/mask_" + file_name + ".png", masked)
                # # '''
                # #     VISUALIZATION OF GROUND TRUTH CODE ENDS HERE
                # # '''

                # '''
                #     VISUALIZATION OF MODIFYING GROUND TRUTH WITH GRABCUT ALGO , COMMENT THIS IF NOT NEEDED
                # '''
                # #Verification of both paths VID and DET
                # img=cv2.imread(root_path_1+img_info["file_name"])
                # if img is None:
                #     print(f"USING PATH 2")
                #     img = cv2.imread(root_path_2 + img_info["file_name"])
                #
                # file_name = img_info["file_name"].split("/")[-1].split(".")[0]
                #
                # #Drawing mask from bbox on image
                # result = cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 5)
                # cv2.imwrite("browse_images/bbox_" + file_name + ".png", result)
                # #
                # # # Now for mask
                # mask_grabcut = np.zeros(img.shape[:2], dtype="uint8")
                # bgdModel = np.zeros((1, 65), np.float64)
                # fgdModel = np.zeros((1, 65), np.float64)
                # rect = (int(x1), int(y1), int(w), int(h))
                #
                # cv2.grabCut(img, mask_grabcut, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
                #
                # # mask2 = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 1).astype('uint8')
                # mask2 = np.where((mask_grabcut == cv2.GC_PR_FGD), 1, 0).astype('uint8')
                #
                # mask_coordinates = np.where(mask2 == cv2.GC_FGD)
                #
                # segmentation_coords = [None]*(len(mask_coordinates[0])+len(mask_coordinates[1]))
                # segmentation_coords[::2] = mask_coordinates[1]
                # segmentation_coords[1::2] = mask_coordinates[0]
                #
                # masked_grabcut = img * mask2[:, :, np.newaxis]
                # # apply our mask -- notice how only the person in the image is
                # # cropped out
                # # masked = cv2.bitwise_and(img, img, mask=mask)
                # cv2.imwrite("browse_images/mask_" + file_name + ".png", masked_grabcut)
                # # # '''
                # # #     VISUALIZATION OF GROUND TRUTH WITH GRABCUT ALGO CODE ENDS HERE
                # # # '''

                if 'instance_id' in ann:
                    gt_instance_ids.append(ann['instance_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # try:
        seg_map = img_info['filename'].replace('jpg', 'png')
        # except:
        # seg_map = img_info['filename'].replace('JPEG', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks, #incase of without mask for testing with faster rcnn
            seg_map=seg_map)

        if self.load_as_video:
            ann['instance_ids'] = np.array(gt_instance_ids).astype(np.int)
        else:
            ann['instance_ids'] = np.arange(len(gt_labels))

        return ann

    def evaluate(self,
                 results,
                 metric=['bbox', 'track'],
                 logger=None,
                 bbox_kwargs=dict(
                     classwise=False,
                     proposal_nums=(100, 300, 1000),
                     iou_thrs=None,
                     metric_items=None),
                 track_kwargs=dict(
                     iou_thr=0.3,
                     ignore_iof_thr=0.5,
                     ignore_by_classes=False,
                     nproc=4)):
        """Evaluation in COCO protocol and CLEAR MOT metric (e.g. MOTA, IDF1).

        Args:
            results (dict): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'track'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            bbox_kwargs (dict): Configuration for COCO styple evaluation.
            track_kwargs (dict): Configuration for CLEAR MOT evaluation.

        Returns:
            dict[str, float]: COCO style and CLEAR MOT evaluation metric.
        """
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['bbox', 'segm', 'track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        eval_results = dict()
        if 'track' in metrics:
            assert len(self.data_infos) == len(results['track_bboxes'])
            inds = [
                i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0
            ]
            num_vids = len(inds)
            inds.append(len(self.data_infos))

            track_bboxes = [
                results['track_bboxes'][inds[i]:inds[i + 1]]
                for i in range(num_vids)
            ]
            ann_infos = [self.get_ann_info(_) for _ in self.data_infos]
            ann_infos = [
                ann_infos[inds[i]:inds[i + 1]] for i in range(num_vids)
            ]
            track_eval_results = eval_mot(
                results=track_bboxes,
                annotations=ann_infos,
                logger=logger,
                classes=self.CLASSES,
                **track_kwargs)
            eval_results.update(track_eval_results)

        # evaluate for detectors without tracker
        super_metrics = ['bbox', 'segm']
        super_metrics = [_ for _ in metrics if _ in super_metrics]
        if super_metrics:
            if isinstance(results, dict):
                if 'bbox' in super_metrics and 'segm' in super_metrics:
                    super_results = []
                    for bbox, mask in zip(results['det_bboxes'],
                                          results['det_masks']):
                        super_results.append((bbox, mask))
                else:
                    super_results = results['det_bboxes']
            elif isinstance(results, list):
                super_results = results
            else:
                raise TypeError('Results must be a dict or a list.')
            super_eval_results = super().evaluate(
                results=super_results,
                metric=super_metrics,
                logger=logger,
                **bbox_kwargs)
            eval_results.update(super_eval_results)

        return eval_results
