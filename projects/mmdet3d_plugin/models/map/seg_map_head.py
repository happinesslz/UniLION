import copy

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import force_fp32
from mmdet.models.utils.transformer import inverse_sigmoid
from torch import nn
import torch.utils.checkpoint as cp

from mmdet3d.core import (
    PseudoSampler,
    xywhr2xyxyr,
)
from mmdet3d.models.builder import HEADS, build_loss
from projects.mmdet3d_plugin.models.utils import draw_heatmap_gaussian, gaussian_radius
from projects.mmdet3d_plugin.models.utils import PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)
from mmdet3d.models import builder

from torch.nn.init import kaiming_normal_
from einops import repeat, rearrange, einsum
import cv2

from projects.mmdet3d_plugin.models.utils.bev_grid_transform import BEVGridTransform


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@HEADS.register_module()
class SegMapHead(nn.Module):
    def __init__(
            self,
            seg_map_config=None,
            in_channels=128 * 3,
            hidden_channel=128,
            num_classes=4,
            # config for Transformer
            num_decoder_layers=3,
            num_heads=8,
            nms_kernel_size=1,
            bn_momentum=0.1,
            bias="auto",
            # others
            train_cfg=None,
            test_cfg=None,
            with_cp=False,
    ):
        super(SegMapHead, self).__init__()

        self.fp16_enabled = False

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_cp = with_cp

        self.seg_map_classes = seg_map_config['seg_map_classes']
        self.transform = BEVGridTransform(**seg_map_config)
        
        bev_backbone = dict(
            type='ResSECOND',
            in_channels=in_channels,
            out_channels=[256, 256, 256],
            blocks_nums=[5, 5, 5],
            layer_strides=[1, 2, 2],
            with_cp= self.with_cp
        )

        pts_neck = dict(
            type='SECONDFPN_CP',
            in_channels=[256, 256, 256],
            out_channels=[256, 256, 256],
            upsample_strides=[1, 2, 4],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True,
            with_cp=self.with_cp
        )

        # a shared convolution
        in_channels =  sum([256, 256, 256])
        self.shared_conv = build_conv_layer(
            dict(type="Conv2d"),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.bev_encoder = nn.Sequential(
            builder.build_backbone(bev_backbone),
            builder.build_neck(pts_neck)
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(True),
            nn.Conv2d(hidden_channel, hidden_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(True),
            nn.Conv2d(hidden_channel, len(self.seg_map_classes), 1),
        )

        self.init_weights()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        if hasattr(self, 'decoder'):
            for m in self.decoder.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward_single(self, inputs):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        inputs = self.bev_encoder(inputs)[0]
        lidar_feat = self.shared_conv(inputs)

        seg_feat = self.transform(lidar_feat)
        pred_seg = self.seg_head(seg_feat)

        res_layer = dict()
        res_layer["pred_seg"] = pred_seg

        return [res_layer]

    def forward(self, feats):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats)
        assert len(res) == 1, "only support one level features."
        return res

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, preds_dicts, metas, gt_bev_masks):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        for index, name in enumerate(self.seg_map_classes):
            loss = sigmoid_focal_loss(preds_dict["pred_seg"][:, index], gt_bev_masks[:, index])
            loss_dict[f"map_loss_seg_{name}"] = loss

        return loss_dict

    def get_bboxes(self, preds_dicts, metas, **data):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """

        pred_seg = preds_dicts[0][0]["pred_seg"].sigmoid()

        seg_ret_layer = []
        for i in range(len(pred_seg)):
            seg = pred_seg[i]
            seg_ret_layer.append(seg)

        return seg_ret_layer, None
