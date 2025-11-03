# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core import multi_apply
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss

from projects.mmdet3d_plugin.models.losses.occ_loss import lovasz_softmax, geo_scal_loss, sem_scal_loss
from projects.mmdet3d_plugin.models.utils.bev_grid_transform import BEVGridTransform


nusc_class_frequencies_dict = dict(
    others=944004,
    barrier=1897170,
    bicycle=152386,
    bus=2391677,
    car=16957802,
    construction_vehicle=724139,
    motorcycle=189027,
    pedestrian=2074468,
    traffic_cone=413451,
    trailer=2384460,
    truck=5916653,
    driveable_surface=175883646,
    other_flat=4275424,
    sidewalk=51393615,
    terrain=61411620,
    manmade=105975596,
    vegetation=116424404,
    free=1892500630
)


flow_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian',
]


@HEADS.register_module()
class BEVOCCHead(BaseModule):
    def __init__(self,
                 in_channels=256,
                 hidden_channel=256,
                 Dz=16,
                 flow=False,
                 bev_occ_config=None,
                 use_mask=True,
                 class_names=None,
                 num_classes=18,
                 size=None,
                 class_balance=False,
                 loss_occ=None,
                 loss_occ_flow=dict(type='L1Loss', loss_weight=0.25),
                 ):
        super(BEVOCCHead, self).__init__()
        self.class_names = class_names
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.Dz = Dz
        self.flow = flow
        self.size = size

        self.final_conv = ConvModule(
            in_channels,
            hidden_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )

        if self.flow:
            self.loss_occ_flow = build_loss(loss_occ_flow)
            self.flow_predicter = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel * 2),
                # nn.ReLU(),
                nn.Linear(hidden_channel * 2, 2 * Dz),
             )
            self.flow_index = list()
            for n in flow_class_names:
                self.flow_index.append(class_names.index(n))

        self.predicter = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel * 2),
            nn.Softplus(),
            nn.Linear(hidden_channel * 2, num_classes * Dz),
        )
        self.use_mask = use_mask
        self.num_classes = num_classes

        self.transform = None
        if bev_occ_config is not None:
            self.transform = BEVGridTransform(**bev_occ_config)

        self.class_balance = class_balance
        if self.class_balance:
            nusc_class_frequencies = list()
            for n in class_names:
                if n in nusc_class_frequencies_dict:
                    nusc_class_frequencies.append(nusc_class_frequencies_dict[n])
                else:
                    nusc_class_frequencies.append(1)
            nusc_class_frequencies = np.array(nusc_class_frequencies)
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
            self.cls_weights = class_weights
        else:
            self.cls_weights = torch.ones(self.num_classes, dtype=torch.float32)
        self.loss_occ = build_loss(loss_occ)

    def forward_single(self, feats):
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        if self.transform is not None:
            feats = self.transform(feats)
        occ_feat = self.final_conv(feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_feat.shape[:3]
        occ_pred = self.predicter(occ_feat)
        occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        if self.flow:
            occ_flow_pred = self.flow_predicter(occ_feat)
            occ_flow_pred = occ_flow_pred.view(bs, Dx, Dy, self.Dz, 2)
        else:
            occ_flow_pred = None
        res_layer = [dict(occ_pred=occ_pred, occ_flow_pred=occ_flow_pred)]
        return res_layer

    def forward(self, feats):
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats)
        assert len(res) == 1, "only support one level features."
        return res

    def loss(self, preds_dicts, metas, gt_occ, mask_camera, gt_occ_flow=None):
        occ_pred = preds_dicts[0][0]['occ_pred']
        loss = dict()
        voxel_semantics = gt_occ.long()

        preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()  # (B, n_cls, Dx, Dy, Dz)
        loss_occ = self.loss_occ(
            preds,
            voxel_semantics,
            weight=self.cls_weights.to(preds),
        ) * 100.0

        occ_weight = 1.0 ### 1.0
        
        loss['occ_loss_voxel_sem_scal'] = sem_scal_loss(preds, voxel_semantics) * occ_weight
        loss['occ_loss_voxel_geo_scal'] = geo_scal_loss(preds, voxel_semantics, non_empty_idx=self.num_classes - 1) * occ_weight
        loss['occ_loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics) * occ_weight

        loss['occ_loss_occ'] = loss_occ * occ_weight

        if self.flow:
            occ_flow_pred = preds_dicts[0][0]['occ_flow_pred']
            occ_flow_pred = occ_flow_pred.reshape(-1, 2)
            gt_occ_flow = gt_occ_flow.reshape(-1, 2)
            visible_mask = torch.zeros_like(occ_flow_pred[:, 0], dtype=torch.bool)
            occ_label = voxel_semantics

            for index in self.flow_index:
                visible_mask |= (occ_label.reshape(-1, 1) == index).reshape(-1).squeeze(-1)

            occ_flow_pred = occ_flow_pred[visible_mask]
            gt_occ_flow = gt_occ_flow[visible_mask]

            gt_occ_weight = torch.norm(gt_occ_flow, dim=-1).unsqueeze(-1)
            gt_occ_weight[gt_occ_weight == 0] = 0.01
            
            
            if occ_flow_pred.numel() > 0:
                loss_occ_flow = self.loss_occ_flow(occ_flow_pred, gt_occ_flow, gt_occ_weight)
            else:
                loss_occ_flow = occ_flow_pred.sum() * 0.0
            
            loss['occ_loss_flow'] = loss_occ_flow * 10.0

        return loss

    def get_bboxes(self, preds_dicts, metas, **data):
        occ_pred = preds_dicts[0][0]['occ_pred']
        occ_score = occ_pred.softmax(-1)  # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)  # (B, Dx, Dy, Dz)

        if self.flow:
            occ_flow_pred = preds_dicts[0][0]['occ_flow_pred']

        ret_layer = []
        for i in range(len(occ_res)):
            occ = occ_res[i]
            if self.flow:
                occ_flow = occ_flow_pred[i]
                # import pdb;pdb.set_trace()
                ret = dict(occ=occ, flow=occ_flow)
            else:
                ret = dict(occ=occ)
            ret_layer.append(ret)

        assert len(ret_layer) == 1

        return ret_layer