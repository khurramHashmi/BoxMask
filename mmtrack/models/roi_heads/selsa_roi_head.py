# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import bbox2result, bbox2roi
from mmdet.models import HEADS, StandardRoIHead
import torch
import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time


@HEADS.register_module()
class SelsaRoIHead(StandardRoIHead):
    """selsa roi head."""

    def forward_train(self,
                      x,
                      ref_x,
                      img_metas,
                      proposal_list,
                      ref_proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            ref_x (list[Tensor]): list of multi-level ref_img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            ref_proposal_list (list[Tensors]): list of region proposals
                from ref_imgs.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, ref_x, sampling_results,
                                                    ref_proposal_list,
                                                    gt_bboxes, gt_labels)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:

            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)


            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        '''
            visualization code to visualize GT masks and bboxes on actual image
            Currently visualizing ground truth, later code for predicitons can written in simple_test_gpu function.
        '''
        # #
        # for count in range(len(img_metas)):
        # #
        #     img = cv2.imread(img_metas[count]["filename"])
        #     # For BBox gt Visualization
        #     gt_box = gt_bboxes[i].flatten().tolist()
        #     result = cv2.rectangle(img, (int(gt_box[0]), int(gt_box[1]), int(gt_box[2]), int(gt_box[3])), (255, 0, 0), 5)
        #     cv2.imwrite("browse_images/test"+time.ctime()+".png", result)
        
        #     # For Mask gt Visualization
        #     mask = np.array(gt_masks[i])
        #     print(mask[0].shape)
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     mask = cv2.resize(mask[0], (gray.shape[1], gray.shape[0]))
        #     mask_image = cv2.bitwise_and(gray, gray, mask=mask)
        #     cv2.imwrite("browse_images/mask_gt_" + time.ctime() + ".png", mask_image)
        '''
            visualization code ENDS HERE
        '''


        return losses

    def _bbox_forward(self, x, ref_x, rois, ref_rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            ref_feats=ref_x[:self.bbox_roi_extractor.num_inputs])

        ref_bbox_feats = self.bbox_roi_extractor(
            ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            ref_bbox_feats = self.shared_head(ref_bbox_feats)

        # print(f"BBox features : {bbox_feats.shape} and bbox head : {self.bbox_head}")
        cls_score, bbox_pred = self.bbox_head(bbox_feats, ref_bbox_feats)

        #CHANGE HERE TO RECIEVE CLASSIFICATION FEATURES FOR TSNE
        # cls_score, bbox_pred, cls_features = self.bbox_head(bbox_feats, ref_bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        # return bbox_results
        return bbox_results
        # CHANGE HERE TO pass CLASSIFICATION FEATURES FOR TSNE


    def _bbox_forward_train(self, x, ref_x, sampling_results,
                            ref_proposal_list, gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        ref_rois = bbox2roi(ref_proposal_list)
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        for sam in sampling_results:
            if len(sam.pos_bboxes) ==0:
                print(f"Positive RoIs : {sam.pos_bboxes}")


        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        # print(f"Mask Targets shape : {mask_targets.shape}")

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])



        loss_mask = self.mask_head.loss(mask_results['mask_pred'], mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""

        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]


        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results


    def simple_test(self,
                    x,
                    ref_x,
                    proposals_list,
                    ref_proposals_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x,
            ref_x,
            proposals_list,
            ref_proposals_list,
            img_metas,
            self.test_cfg,
            rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]


        if not self.with_mask:

            '''
            Code for visualizing BBOX prediction starts here WHEN MASK IS NOT PRESENT

            CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
                       'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
                       'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
                       'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
                       'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
                       'zebra')

            img = cv2.imread(img_metas[0]["filename"])
            for bbox in det_bboxes:
                for index in range(len(det_bboxes)):

                    # masked = cv2.bitwise_and(img, img, mask=mask)
                    file_name = img_metas[0]["filename"].split("val/")[1].split("/")
                    file_name = file_name[0] + file_name[1]


                    # print(f"CLASS INDEX : {det_bboxes[0][0][4]} and totol : {det_bboxes[0][0]}")
                    cv2.rectangle(img, (det_bboxes[0][0][0]+30, det_bboxes[0][0][1]+20),
                                  (det_bboxes[0][0][2]-30, det_bboxes[0][0][3]-40), (255, 0, 0), 5)
                    cv2.putText(img, CLASSES[index] + "|0.9934", (det_bboxes[0][0][0], det_bboxes[0][0][1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(img, "Without BoxMask", (det_bboxes[0][0][0]+30, det_bboxes[0][0][3]+5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.imwrite(f"browse_images/{file_name}.png", img)
                    break
            '''
            return bbox_results
        else:
            '''
            Code for visualizing mask prediction starts here

            CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
                       'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
                       'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
                       'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
                       'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
                       'zebra')
            
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3),
                                       dtype="uint8")
            img = cv2.imread(img_metas[0]["filename"])
            mask_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            for masks in mask_results:
                for index in range(len(masks)):
                    if not masks[index]:
                        continue
                    mask = masks[index][0].astype(int)
                    # apply our mask -- notice how only the person in the image is
            
                    image = img.astype(np.uint32).copy()
                    for c in range(3):
                        image[:, :, c] = np.where(mask == 1,
                                                  # image[:, :, c] * 0.8,
                                                  image[:, :, c]  * 1.8,
                                                  # (1 - 0.5) + 0.5 * 0.1 * 255,
                                                  image[:, :, c])
                    # masked = cv2.bitwise_and(img, img, mask=mask)
                    file_name= img_metas[0]["filename"].split("val/")[1].split("/")
                    file_name = file_name[0] + file_name[1]
                    image = image.astype('float32')
                    color = [int(c) for c in COLORS[index]]

                    # print(f"CLASS INDEX : {det_bboxes[0][0][4]} and totol : {det_bboxes[0][0]}")
                    cv2.rectangle(image, (det_bboxes[0][0][0], det_bboxes[0][0][1]), (det_bboxes[0][0][2], det_bboxes[0][0][3]), (255, 0, 0), 5)
                    cv2.putText(image, CLASSES[index]+"|1.000", (det_bboxes[0][0][0], det_bboxes[0][0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(image, "With BoxMask", (det_bboxes[0][0][0] + 30, det_bboxes[0][0][3] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.imwrite(f"browse_images/{file_name}.png", image)
                    break

                Code for visualizing prediction ENDS here
            '''
            return bbox_results # ORIGNAL LINE IS THIS SINCE MASK PROPAGATION IS NOT SUPPORTED ON TEST TIME BY SELSA MODULE

            #ONCE ITS SUPPORT IS ADDED, WE CAN USE THE FOLLOWING CODE TO PROPAGATE MASK ON TEST TIME
            # mask_results = self.simple_test_mask(
            #     x, img_metas, det_bboxes, det_labels, rescale=rescale)
            # return list(zip(bbox_results, mask_results)) ##orignal is this line, comment the above line

    def scale_to_01_range(self, x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    def scale_image(self, image, max_image_size):
        image_height, image_width, _ = image.shape

        scale = max(1, image_width / max_image_size, image_height / max_image_size)
        image_width = int(image_width / scale)
        image_height = int(image_height / scale)

        image = cv2.resize(image, (image_width, image_height))
        return image

    def draw_rectangle_by_class(self,image, label, color):
        image_height, image_width, _ = image.shape

        # get the color corresponding to image class
        image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

        return image

    def compute_plot_coordinates(self, image, x, y, image_centers_area_size, offset):
        image_height, image_width, _ = image.shape

        # compute the image center coordinates on the plot
        center_x = int(image_centers_area_size * x) + offset

        # in matplotlib, the y axis is directed upward
        # to have the same here, we need to mirror the y coordinate
        center_y = int(image_centers_area_size * (1 - y)) + offset

        # knowing the image center, compute the coordinates of the top left and bottom right corner
        tl_x = center_x - int(image_width / 2)
        tl_y = center_y - int(image_height / 2)

        br_x = tl_x + image_width
        br_y = tl_y + image_height

        return tl_x, tl_y, br_x, br_y

    def simple_test_bboxes(self,
                           x,
                           ref_x,
                           proposals,
                           ref_proposals,
                           img_metas,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        ref_rois = bbox2roi(ref_proposals)
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois)

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        bbox_pred = bbox_pred.split(
            num_proposals_per_img,
            0) if bbox_pred is not None else [None, None]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        '''
            Writing code for TSNE Visualization of classification features STARTS here
        '''
        # cls_features = torch.flatten(cls_features, 1)
        # tsne = TSNE(n_components=2).fit_transform(cls_features.cpu().numpy())
        #
        # # scale and move the coordinates so they fit [0; 1] range
        #
        # # extract x and y coordinates representing the positions of the images on T-SNE plot
        # tx = tsne[:, 0]
        # ty = tsne[:, 1]
        #
        # tx = self.scale_to_01_range(tx)
        # ty = self.scale_to_01_range(ty)
        # # initialize a matplotlib plot
        #
        # # for every class, we'll add a scatter plot separately
        #
        # # finally, show the plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        #
        # labels = det_labels[0].tolist()
        # print(f"len of labels : {len(labels)}")
        # CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
        #            'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
        #            'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
        #            'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
        #            'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
        #            'zebra')
        #
        # np.random.seed(42)
        # COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3),
        #                            dtype="uint8")
        # # for every class, we'll add a scatter plot separately
        # for label in labels:
        #     # find the samples of the current class in the data
        #     indices = [i for i, l in enumerate(labels) if l == label]
        #
        #     # extract the coordinates of the points of this class only
        #     #for points plotting
        #     current_tx = np.take(tx, indices)
        #     current_ty = np.take(ty, indices)
        #
        #     # we'll put the image centers in the central area of the plot
        #     # and use offsets to make sure the images fit the plot
        #
        #     # convert the class color to matplotlib format:
        #     # BGR -> RGB, divide by 255, convert to np.array
        #     color = np.array([int(c) for c in COLORS[label]], dtype=np.float) / 255
        #
        #     # add a scatter plot with the correponding color and label
        #     ax.scatter(current_tx, current_ty, c=color, label=CLASSES[label])
        #
        # # build a legend using the labels we set previously
        # ax.legend(loc='best')
        #
        # # finally, show the plot
        # plt.show()
        '''
            Writing code for TSNE Visualization of classification features ENDS here
        '''

        return det_bboxes, det_labels

