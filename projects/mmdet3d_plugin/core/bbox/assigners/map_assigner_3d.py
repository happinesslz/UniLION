import torch
import torch.nn.functional as F
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class MapAssigner(BaseAssigner):
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 pts_cost=dict(type='ChamferDistance', loss_src_weight=1.0, loss_dst_weight=1.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.pts_cost = build_match_cost(pts_cost)
        self.pc_range = pc_range

    def assign(self,
               pts_pred,
               cls_pred,
               gt_labels,
               gt_pts,
               gt_bboxes_ignore=None,
               eps=1e-7):

        num_gts, num_bboxes = gt_pts.size(0), pts_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = pts_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = pts_pred.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred[0].T, gt_labels)
        # regression L1 cost

        _, num_orders, num_pts_per_gtline, num_coords = gt_pts.shape

        num_pts_per_predline = pts_pred.size(1)

        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(pts_pred.permute(0, 2, 1), size=(num_pts_per_gtline),
                                                  mode='linear', align_corners=True)
            pts_pred_interpolated = pts_pred_interpolated.permute(0, 2, 1).contiguous()
        else:
            pts_pred_interpolated = pts_pred
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_cost_ordered = self.pts_cost(pts_pred_interpolated, gt_pts)
        pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)

        # weighted sum of above three costs
        cost = cls_cost + pts_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            pts_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            pts_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels), order_index