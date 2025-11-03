# modified from https://github.com/Haiyang-W/DSVT
import torch
import torch.nn as nn
from mmdet3d.models import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class HeightCompression(nn.Module):
    def __init__(self, output_shape, num_bev_feats, **kwargs):
        super().__init__()
        self.nx, self.ny, self.nz = output_shape
        self.num_bev_feats = num_bev_feats
        self.num_bev_feats_ori = num_bev_feats // self.nz

    def forward(self, voxel_features):
        spatial_features = voxel_features.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 2, 3, 4).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features
