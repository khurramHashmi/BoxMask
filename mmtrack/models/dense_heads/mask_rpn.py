# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32, ModuleList

from mmtrack.models.dense_heads.mask_anchor_head import MaskAnchorHead
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.core import images_to_levels, multi_apply, bbox2roi



@HEADS.register_module()
class MaskRPNHead(MaskAnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 num_convs=1,
                 conv_out_channels=512,
                 conv_kernel_size=3,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 predictor_cfg=dict(type='Conv'),
                 mask_roi_extractor=None,
                 mask_head=None,
                 **kwargs):
        self.num_convs = num_convs

        super(MaskRPNHead, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)

        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_cfg = upsample_cfg
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.predictor_cfg = predictor_cfg
        self.upsample_method = self.upsample_cfg.get('type')

        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        # if mask_head is not None:
        #     self.init_mask_head(mask_roi_extractor, mask_head)

        '''
        Creating Mask network module in RPN similar to the mask module used in FCN Mask head but with single Convolution
        '''
        self.mask_convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.mask_convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()

        upsample_cfg_.update(
            in_channels=upsample_in_channels,
            out_channels=self.conv_out_channels,
            kernel_size=self.scale_factor,
            stride=self.scale_factor)
        self.upsample = build_upsample_layer(upsample_cfg_)

        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        #The output for rpn conv_logits will be one since we are just estimating background or object
        self.rpn_conv_logits = build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, self.num_base_priors * self.cls_out_channels, 1)

        self.relu = nn.ReLU(inplace=True)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4,
                                 1)



    def forward_single(self, x):
        """Forward feature map of a single scale level."""

        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)

        # print(f"After rpn_bbox_pred shape : {rpn_bbox_pred.shape}")
        return rpn_cls_score, rpn_bbox_pred

    def rpn_mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        '''
            Predicting mask on the basis of bbox predictions executed by the RPN
        '''
        # rois_lis = []
        # rois_lis.append(rois)
        # print(f"Input features for mask network : {x[0].shape}")

        # print(rois, pos_inds, bbox_feats)
        #RoIs need to be in the right format as RoI must be (idx, x1, y1, x2, y2)
        # print(rois)
        rois = bbox2roi(rois)

        '''
        Checking BBOX to ROI function code that where problem exists instead of directly calling it
        '''
        # rois_list = []
        # for img_id, bboxes in enumerate(rois_lis):
        #     print(img_id, bboxes)
        #     if bboxes.size(0) > 0:
        #         img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        #         rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        #     else:
        #         rois = bboxes.new_zeros((0, 5))
        #     rois_list.append(rois)
        # rois = torch.cat(rois_list, 0)
        '''
        Code checking ends here.
        '''

        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))

        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            # print(f"MASK ROI FEATURE NAME : {self.mask_roi_extractor} AND TENSOR {mask_feats.shape}")
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        # mask_in_features =
        for conv in self.mask_convs:
            mask_features = conv(mask_feats)
        if self.upsample is not None:
            mask_features = self.upsample(mask_feats)
            if self.upsample_method == 'deconv':
                mask_features = self.relu(mask_features)


        # print(f"Output features after mask FCN : {mask_features.shape}")


        rpn_mask_pred = self.rpn_conv_logits(mask_features)
        # print(f"RPN Mask Pred : {rpn_mask_pred.shape}")
        return rpn_mask_pred

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, mask_preds, mask_targets, num_total_samples, mask_pos_assigned_gt_inds):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        # print(f"CLs pred shape : {cls_score.shape} and Cls target shape {labels.shape} and lebels weight shape {label_weights.shape}")
        # if mask_preds[0].ndim > 3:
        #     print(f"THIS ONE WORKING HERE")
        #     mask_preds[0] = mask_preds[0].permute(1, 0, 2, 3)
        #     print(f"SHAPE NOW : {mask_preds[0].shape}")
        #     mask_preds[0] = mask_preds[0].view(mask_preds[0].size(1), mask_preds[0].size(2),mask_preds[0].size(3))
        #
        #     mask_targets[0] = mask_targets[0].view(mask_targets[0].size(1), mask_targets[0].size(2), mask_targets[0].size(3))
        #     print(f"SHAPE mask_targets : {mask_targets[0].shape}")
        # else:
        #     mask_targets[0] = mask_targets[0].view(mask_targets[0].size(1), mask_targets[0].size(2))
        # # mask_preds[0] = mask_preds[0].view(mask_preds[0].size(0), mask_preds[0].size(2), mask_preds[0].size(3))
        #
        # print(f"after mask_preds {mask_preds[0].shape} and mask_targets {mask_targets[0].shape}")

        # print(f" cls_scores :{cls_score.shape}  cls_labels shape: {labels.shape} and LABELS VALUE WHERE IT IS GREATER THAN 1 {labels[torch.where(labels > 1)]}")

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        # print(f"BBox pred shape : {bbox_pred.shape} and BBOX target shape {bbox_targets.shape}")
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        # print(f"AFTER BBox pred shape : {bbox_pred.shape} and BBOX target shape {bbox_targets.shape}")
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        # print(f"{bbox_pred.shape} {bbox_targets.shape} {bbox_weights.shape} {num_total_samples}")
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        #To compute mask loss, we need mask_targets and mask_weights
        # print(f"Mask Prediction shape :·{mask_preds.shape} and mask_targets shape {mask_targets.shape}and mask_pos_assigned_gt_inds shape {mask_pos_assigned_gt_inds[0]}")

        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            # if self.class_agnostic:
            loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(mask_pos_assigned_gt_inds[0]))
            # else:
            # loss_mask=self.loss_mask(mask_preds, mask_targets, mask_pos_assigned_gt_inds[0])


        return loss_cls, loss_bbox, loss_mask

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels=None,
             img_metas=None,
             gt_masks=None,
             gt_bboxes_ignore=None,
             input_features=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # losses = super(MaskRPNHead, self).loss(
        #     cls_scores,
        #     bbox_preds,
        #     gt_bboxes,
        #     None,
        #     img_metas,
        #     gt_bboxes_ignore=gt_bboxes_ignore)
        # return dict(
        #     loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])


        '''
            Modified loss function for MaskRPN 
        '''
        mask_preds = None

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        # print(f"gt_labels : {gt_labels} img_metas : {img_metas} gt_masks : {gt_masks} gt_bboxes {gt_bboxes} input_features {input_features[0].shape}")

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_masks_list=gt_masks,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            return_sampling_results=True
        )
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, mask_targets_list, sampling_results_list) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)

        '''
            Predcting Mask here since pos_bboxes, post_gt_inds are present here, later this code should be organized
        '''
        # print(f"Sampling results : {sampling_results_list}")

        mask_pos_proposals = [res.pos_bboxes for res in sampling_results_list]
        mask_pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results_list
        ]
        # print(f"mask_pos_assigned_gt_inds : {mask_pos_assigned_gt_inds}")
        mask_preds = self.rpn_mask_forward(input_features, rois=mask_pos_proposals, pos_inds=mask_pos_assigned_gt_inds, bbox_feats=None)
        # print(f"mask_pred SHAPE  {mask_preds.shape}")

        '''
            Predcting Mask Code ends here 
        '''


        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        

        losses_cls, losses_bbox, losses_mask = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mask_preds=mask_preds,
            mask_targets=mask_targets_list,
            num_total_samples=num_total_samples,
            mask_pos_assigned_gt_inds=mask_pos_assigned_gt_inds)

        # print(f"loss_cls :·{losses_cls} and loss_bbox {losses_bbox} and loss_mask {losses_mask}")

        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox, loss_rpn_mask=losses_mask)


    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_anchors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            # print(f"RPN CLS SCORES BEFORE SIGMOID : {rpn_cls_score.shape}")
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            # print(f"RPN CLS SCORES AFTER SIGMOID : {scores.shape}")
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def onnx_export(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            Tensor: dets of shape [N, num_det, 5].
        """
        cls_scores, bbox_preds = self(x)

        assert len(cls_scores) == len(bbox_preds)

        batch_bboxes, batch_scores = super(MaskRPNHead, self).onnx_export(
            cls_scores, bbox_preds, img_metas=img_metas, with_nms=False)
        # Use ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        cfg = copy.deepcopy(self.test_cfg)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        # Different from the normal forward doing NMS level by level,
        # we do NMS across all levels when exporting ONNX.
        dets, _ = add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                         cfg.max_per_img,
                                         cfg.nms.iou_threshold,
                                         score_threshold, nms_pre,
                                         cfg.max_per_img)
        return dets