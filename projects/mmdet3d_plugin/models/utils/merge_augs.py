# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.ops import nms_rotated, box_iou_rotated
from mmdet3d.core.bbox import bbox3d2result, bbox3d_mapping_back, xywhr2xyxyr
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)

import pickle
import mmcv

from mmcv.ops import boxes_iou_bev

ensemble = False


def merge_aug_bboxes_3d(aug_results, data_list, test_cfg):
    recovered_bboxes = []
    recovered_scores = []
    recovered_labels = []

    for bboxes, data in zip(aug_results, data_list):
        bda_cfg = data['img_metas'][0]['bda_cfg']
        scale_factor = bda_cfg['scale_bda']
        pcd_horizontal_flip = bda_cfg['flip_dy']
        pcd_vertical_flip = bda_cfg['flip_dx']
        recovered_scores.append(bboxes['scores_3d'])
        recovered_labels.append(bboxes['labels_3d'])
        bboxes = bbox3d_mapping_back(bboxes['boxes_3d'], scale_factor, pcd_horizontal_flip, pcd_vertical_flip)
        recovered_bboxes.append(bboxes)

    aug_bboxes = recovered_bboxes[0].cat(recovered_bboxes)
    aug_bboxes_for_nms = aug_bboxes.bev
    aug_scores = torch.cat(recovered_scores, dim=0)
    aug_labels = torch.cat(recovered_labels, dim=0)

    test_cfg = test_cfg.copy()
    ################ extra added
    test_cfg['nms_type'] = 'rotate'
    test_cfg['use_rotate_nms'] = True
    test_cfg['max_num'] = 500
    test_cfg['nms_thr'] = 0.1
    test_cfg['score_threshold'] = 0.05

    merged_bboxes = []
    merged_scores = []
    merged_labels = []

    # Apply multi-class nms when merge bboxes
    if len(aug_labels) == 0:
        return bbox3d2result(aug_bboxes, aug_scores, aug_labels)

    for class_id in range(torch.max(aug_labels).item() + 1):
        class_inds = (aug_labels == class_id)
        bboxes_i = aug_bboxes[class_inds]
        bboxes_nms_i = aug_bboxes_for_nms[class_inds, :]
        scores_i = aug_scores[class_inds]
        labels_i = aug_labels[class_inds]
        if len(bboxes_nms_i) == 0:
            continue

        selected = nms_rotated(bboxes_nms_i, scores_i, test_cfg.nms_thr)[1]

        if True:  # voting
            vote_iou_thresh = 0.65
            use_voting_scores = False

            selected_bboxes = bboxes_i[selected, :]
            selected_scores = scores_i[selected]
            selected_labels = labels_i[selected]

            iou = box_iou_rotated(selected_bboxes.bev, bboxes_nms_i)
            iou[iou < vote_iou_thresh] = 0.

            voted_bboxes = (iou[:, :, None] * bboxes_i.tensor[None]).sum(dim=1) / (iou[:, :, None].sum(dim=1) + 1e-6)
            voted_bboxes[:, 6] = torch.atan2(
                (iou * torch.sin(bboxes_i.tensor[None, :, 6])).sum(dim=1) / (iou.sum(dim=1) + 1e-6),
                (iou * torch.cos(bboxes_i.tensor[None, :, 6])).sum(dim=1) / (iou.sum(dim=1) + 1e-6))

            voted_bboxes = LiDARInstance3DBoxes(voted_bboxes, box_dim=voted_bboxes.shape[-1])

            selected_bboxes = voted_bboxes
            if use_voting_scores:
                voted_scores = (iou * scores_i[None]).sum(dim=1) / iou.sum(dim=1)
                selected_scores = voted_scores

        merged_bboxes.append(selected_bboxes)
        merged_scores.append(selected_scores)
        merged_labels.append(selected_labels)

    merged_bboxes = merged_bboxes[0].cat(merged_bboxes)
    merged_scores = torch.cat(merged_scores, dim=0)
    merged_labels = torch.cat(merged_labels, dim=0)

    _, order = merged_scores.sort(0, descending=True)
    num = min(test_cfg.max_num, len(aug_bboxes))
    order = order[:num]

    merged_bboxes = merged_bboxes[order]
    merged_scores = merged_scores[order]
    merged_labels = merged_labels[order]

    return bbox3d2result(merged_bboxes, merged_scores, merged_labels)