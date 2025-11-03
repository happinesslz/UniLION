import torch
import torch.nn.functional as F
from mmdet3d.core import bbox_overlaps_3d


def calculate_iou_loss(iou_preds, batch_box_preds, gt_boxes, weights):
    """
    Args:
        iou_preds: (batch x 1 x proposal)
        batch_box_preds: (batch x proposal x 7)
        # gt_boxes: (batch x N, 7 or 9)
        gt_boxes: (batch x proposal x 7)
        weights:
    Returns:
    """
    iou = bbox_overlaps_3d(batch_box_preds.reshape(-1, 7), gt_boxes.reshape(-1, 7), 'iou', 'lidar')
    iou_target = torch.diag(iou)
    valid_index = torch.nonzero(iou_target * weights[:, :, 0].view(-1)).squeeze(-1)
    num_pos = valid_index.shape[0]

    iou_target = iou_target * 2 - 1  # [0, 1] ==> [-1, 1]

    loss = F.l1_loss(iou_preds.view(-1)[valid_index], iou_target[valid_index], reduction='sum')
    loss = loss / max(num_pos, 1)
    return loss