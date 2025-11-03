import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class MapBBoxCoder(BaseBBoxCoder):
    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 score_threshold=None,
                 post_center_range=None):
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold

    def encode(self, dst_pts):
        targets = torch.zeros_like(dst_pts)
        # targets[..., 0] = (dst_pts[..., 0] - self.pc_range[0]) / (self.out_size_factor * self.voxel_size[0])
        # targets[..., 1] = (dst_pts[..., 1] - self.pc_range[1]) / (self.out_size_factor * self.voxel_size[1])
        targets[..., 0] = (dst_pts[..., 0] - self.pc_range[0]) / (self.pc_range[2] - self.pc_range[0])
        targets[..., 1] = (dst_pts[..., 1] - self.pc_range[1]) / (self.pc_range[3] - self.pc_range[1])
        return targets

    def bev_encode(self, dst_pts):
        targets = torch.zeros_like(dst_pts)
        targets[..., 0] = (dst_pts[..., 0] - self.pc_range[0]) / (self.out_size_factor * self.voxel_size[0])
        targets[..., 1] = (dst_pts[..., 1] - self.pc_range[1]) / (self.out_size_factor * self.voxel_size[1])
        # targets[..., 0] = (dst_pts[..., 0] - self.pc_range[0]) / (self.pc_range[2] - self.pc_range[0])
        # targets[..., 1] = (dst_pts[..., 1] - self.pc_range[1]) / (self.pc_range[3] - self.pc_range[1])
        return targets

    def decode(self, heatmap, center, filter=False):
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # change size to real world metric
        center[..., 0] = center[..., 0] * (self.pc_range[2] - self.pc_range[0]) + self.pc_range[0]
        center[..., 1] = center[..., 1] * (self.pc_range[3] - self.pc_range[1]) + self.pc_range[1]

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            pts3d = center[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pts': pts3d,
                'scores': scores,
                'labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heatmap.device)
            mask = (center[..., :2] >=
                    self.post_center_range[:2]).all(2).all(2)
            mask &= (center[..., :2] <=
                     self.post_center_range[2:]).all(2).all(2)

            predictions_dicts = []
            for i in range(heatmap.shape[0]):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                pts3d = center[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'pts': pts3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts