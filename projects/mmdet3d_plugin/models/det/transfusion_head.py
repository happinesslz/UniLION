import copy

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.core import (
    PseudoSampler,
    xywhr2xyxyr, circle_nms, LiDARInstance3DBoxes,
)
from mmdet3d.models.builder import HEADS, build_loss

from projects.mmdet3d_plugin.models.losses.det_loss import calculate_iou_loss
from projects.mmdet3d_plugin.models.utils import draw_heatmap_gaussian, gaussian_radius
from projects.mmdet3d_plugin.models.utils import PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)

from torch.nn.init import kaiming_normal_
import torch.utils.checkpoint as cp


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class SeparateHead(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels, num_conv = self.sep_head_dict[cur_name]

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'heatmap' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


@HEADS.register_module()
class TransFusionHead(nn.Module):
    def __init__(
            self,
            num_proposals=128,
            auxiliary=True,
            in_channels=128 * 3,
            hidden_channel=128,
            num_classes=4,
            iou_rescore_weight=0.5,
            # config for Transformer
            num_decoder_layers=3,
            num_heads=8,
            nms_kernel_size=1,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation="relu",
            # config for FFN
            common_heads=dict(),
            num_heatmap_convs=2,
            bias="auto",
            # loss
            loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
            loss_iou=dict(
                type="VarifocalLoss", use_sigmoid=True, iou_weighted=True, reduction="mean"
            ),
            loss_bbox=dict(type="L1Loss", reduction="mean"),
            loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean"),
            # others
            train_cfg=None,
            test_cfg=None,
            bbox_coder=None,
            with_cp=False,
    ):
        super(TransFusionHead, self).__init__()

        self.fp16_enabled = False

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.iou_rescore_weight = iou_rescore_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False
        self.with_cp = with_cp
        
        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type="Conv2d"),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
            )
        )
        layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(
            hidden_channel,
            num_heads,
            ffn_channel,
            dropout,
            activation,
            self_posembed=PositionEmbeddingLearned(2, hidden_channel),
            cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
        )

        # Prediction Head
        heads = copy.deepcopy(common_heads)
        heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
        self.prediction_head = SeparateHead(hidden_channel, 64, 1, heads, use_bias=True)

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

        self.query_radius = 20
        self.query_range = torch.arange(-self.query_radius, self.query_radius + 1)
        self.query_r_coor_x, self.query_r_coor_y = torch.meshgrid(self.query_range, self.query_range)

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

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        inputs = inputs.permute(0, 1, 3, 2).contiguous()
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        x_grid, y_grid = heatmap.shape[-2:]

        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg["dataset"] == "nuScenes":
            local_max[
            :,
            8,
            ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[
            :,
            9,
            ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            local_max[
            :,
            1,
            ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[
            :,
            2,
            ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top #num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
                        ..., : self.num_proposals
                        ]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
            0, 2, 1
        )
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        # compute local key
        top_proposals_x = top_proposals_index // x_grid  # bs, num_proposals
        top_proposals_y = top_proposals_index % y_grid  # bs, num_proposals

        # bs, num_proposal, radius * 2 + 1, radius * 2 + 1
        top_proposals_key_x = top_proposals_x[:, :, None, None] + self.query_r_coor_x[None, None, :, :].to(
            top_proposals.device)
        top_proposals_key_y = top_proposals_y[:, :, None, None] + self.query_r_coor_y[None, None, :, :].to(
            top_proposals.device)
        # bs, num_proposals, key_num
        top_proposals_key_index = top_proposals_key_x.view(batch_size, top_proposals_key_x.shape[1],
                                                           -1) * x_grid + top_proposals_key_y.view(batch_size,
                                                                                                   top_proposals_key_y.shape[
                                                                                                       1], -1)
        key_mask = (top_proposals_key_index < 0) + (top_proposals_key_index >= (x_grid * y_grid))
        top_proposals_key_index = torch.clamp(top_proposals_key_index, min=0, max=x_grid * y_grid - 1)
        num_proposals = top_proposals_key_index.shape[1]
        key_feat = lidar_feat_flatten.gather(
            index=top_proposals_key_index.view(batch_size, 1, -1).expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
        key_feat = key_feat.view(batch_size, lidar_feat_flatten.shape[1], num_proposals, -1)
        key_pos = bev_pos.gather(
            index=top_proposals_key_index.view(batch_size, 1, -1).permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1)
        key_pos = key_pos.view(batch_size, num_proposals, -1, bev_pos.shape[-1])
        key_feat = key_feat.permute(0, 2, 1, 3).reshape(batch_size * num_proposals, lidar_feat_flatten.shape[1], -1)
        key_pos = key_pos.view(-1, key_pos.shape[2], key_pos.shape[-1])
        key_padding_mask = key_mask.view(-1, key_mask.shape[-1])

        query_feat_T = query_feat.permute(0, 2, 1).reshape(batch_size * num_proposals, -1, 1)
        query_pos_T = query_pos.view(-1, 1, query_pos.shape[-1])
        
        if self.with_cp:
            query_feat_T = cp.checkpoint(self.decoder,query_feat_T, key_feat,query_pos_T, key_pos, key_padding_mask, None)
            # query_feat_T = self.decoder(
            #     query_feat_T, key_feat, query_pos_T, key_pos, key_padding_mask
            # )  ### (self, query, key, query_pos, key_pos, key_padding_mask=None, attn_mask=None):
        else:
            query_feat_T = self.decoder(
                query_feat_T, key_feat, query_pos_T, key_pos, key_padding_mask
            )
        query_feat = query_feat_T.reshape(batch_size, num_proposals, 128).permute(0, 2, 1)

        res_layer = self.prediction_head(query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
        res_layer["query_feat"] = query_feat

        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        res_layer["dense_heatmap"] = dense_heatmap
        res_layer["query_labels"] = self.query_labels

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

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx: batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            list_of_pred_dict,
            np.arange(len(gt_labels_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        assigned_gt_inds = list(res_tuple[8])
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
            assigned_gt_inds
        )

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict["center"].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height, vel
        )  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]["bboxes"]
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                                  self.num_proposals * idx_layer: self.num_proposals * (idx_layer + 1), :
                                  ]
            score_layer = score[
                          ...,
                          self.num_proposals * idx_layer: self.num_proposals * (idx_layer + 1),
                          ]

            if self.train_cfg.assigner.type == "HungarianAssigner3D":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == "HeuristicAssigner":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])
        feature_map_size = (
                grid_size[:2] // self.train_cfg["out_size_factor"]
        )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(
            self.num_classes, feature_map_size[1], feature_map_size[0]
        )
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
            length = length / voxel_size[1] / self.train_cfg["out_size_factor"]
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                )
                radius = max(self.train_cfg["min_radius"], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                        (x - pc_range[0])
                        / voxel_size[0]
                        / self.train_cfg["out_size_factor"]
                )
                coor_y = (
                        (y - pc_range[1])
                        / voxel_size[1]
                        / self.train_cfg["out_size_factor"]
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                # NOTE: fix
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                )

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)

        ass_inds = torch.stack([pos_inds, sampling_result.pos_assigned_gt_inds], axis=-1)

        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
            ass_inds[None]
        )

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, preds_dicts, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
            assigned_gt_inds
        ) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if hasattr(self, "on_the_image_mask"):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict["dense_heatmap"]),
            heatmap,
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict["det_loss_heatmap"] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False
            ):
                prefix = "layer_-1"
            else:
                prefix = f"layer_{idx_layer}"

            layer_labels = labels[
                           ...,
                           idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                           ].reshape(-1)
            layer_label_weights = label_weights[
                                  ...,
                                  idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                                  ].reshape(-1)
            layer_score = preds_dict["heatmap"][
                          ...,
                          idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                          ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score,
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict["center"][
                           ...,
                           idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                           ]
            layer_height = preds_dict["height"][
                           ...,
                           idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                           ]
            layer_rot = preds_dict["rot"][
                        ...,
                        idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                        ]
            layer_dim = preds_dict["dim"][
                        ...,
                        idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                        ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot], dim=1
            ).permute(
                0, 2, 1
            )  # [BS, num_proposals, code_size]
            if "vel" in preds_dict.keys():
                layer_vel = preds_dict["vel"][
                            ...,
                            idx_layer
                            * self.num_proposals: (idx_layer + 1)
                                                  * self.num_proposals,
                            ]
                preds = torch.cat(
                    [layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1
                ).permute(
                    0, 2, 1
                )  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get("code_weights", None)
            layer_bbox_weights = bbox_weights[
                                 :,
                                 idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                                 :,
                                 ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(
                code_weights
            )
            layer_bbox_targets = bbox_targets[
                                 :,
                                 idx_layer * self.num_proposals: (idx_layer + 1) * self.num_proposals,
                                 :,
                                 ]
            layer_loss_bbox = self.loss_bbox(
                preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1)
            )

            loss_dict[f"det_{prefix}_loss_cls"] = layer_loss_cls
            loss_dict[f"det_{prefix}_loss_bbox"] = layer_loss_bbox

            if "iou" in preds_dict.keys():
                bbox_targets_for_iou = layer_bbox_targets.permute(0, 2, 1)
                rot_iou = bbox_targets_for_iou[:, 6:8, :].clone()
                rot_iou = torch.atan2(rot_iou[:, 0:1, :], rot_iou[:, 1:2, :])
                dim_iou = bbox_targets_for_iou[:, 3:6, :].clone().exp()
                height_iou = bbox_targets_for_iou[:, 2:3, :].clone()
                center_iou = bbox_targets_for_iou[:, 0:2, :].clone()

                feature_map_stride = self.train_cfg["out_size_factor"]
                pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
                voxel_size = torch.tensor(self.train_cfg["voxel_size"])

                center_iou[:, 0, :] = center_iou[:, 0, :] * feature_map_stride * voxel_size[0] + pc_range[0]
                center_iou[:, 1, :] = center_iou[:, 1, :] * feature_map_stride * voxel_size[1] + pc_range[1]
                batch_box_targets_for_iou = torch.cat([center_iou, height_iou, dim_iou, rot_iou], dim=1).permute(0, 2,
                                                                                                                 1)

                rot_pred = layer_rot.clone()
                center_pred = layer_center.clone()
                height_pred = layer_height.clone()
                rot_pred = torch.atan2(rot_pred[:, 0:1, :], rot_pred[:, 1:2, :])
                dim_pred = layer_dim.clone().exp()
                center_pred[:, 0, :] = center_pred[:, 0, :] * feature_map_stride * voxel_size[0] + pc_range[0]
                center_pred[:, 1, :] = center_pred[:, 1, :] * feature_map_stride * voxel_size[1] + pc_range[1]
                batch_box_preds = torch.cat([center_pred, height_pred, dim_pred, rot_pred], dim=1).permute(0, 2, 1)

                batch_box_preds_for_iou = batch_box_preds.clone().detach()
                batch_box_targets_for_iou = batch_box_targets_for_iou.detach()
                layer_iou_loss = calculate_iou_loss(
                    iou_preds=preds_dict['iou'],
                    batch_box_preds=batch_box_preds_for_iou,
                    gt_boxes=batch_box_targets_for_iou,
                    weights=layer_bbox_weights,
                )
                loss_dict[f"det_{prefix}_loss_iou"] = layer_iou_loss * self.iou_rescore_weight

        # loss_dict[f"det_matched_ious"] = layer_loss_cls.new_tensor(matched_ious)
        assigned_gt_inds = [gt_inds[0] for gt_inds in assigned_gt_inds]
        return loss_dict, assigned_gt_inds

    def get_bboxes(self, preds_dicts, outs_planning, metas, **data):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]["heatmap"].shape[0]
            batch_score = preds_dict[0]["heatmap"][..., -self.num_proposals:].sigmoid()

            one_hot = F.one_hot(
                self.query_labels, num_classes=self.num_classes
            ).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0]["query_heatmap_score"] * one_hot

            batch_iou = (preds_dict[0]['iou'][..., -self.num_proposals:] + 1) * 0.5 if 'iou' in preds_dict[0] else None
            batch_center = preds_dict[0]["center"][..., -self.num_proposals:]
            batch_height = preds_dict[0]["height"][..., -self.num_proposals:]
            batch_dim = preds_dict[0]["dim"][..., -self.num_proposals:]
            batch_rot = preds_dict[0]["rot"][..., -self.num_proposals:]
            batch_vel = None
            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"][..., -self.num_proposals:]

            with_motion = outs_planning is not None

            if with_motion:
                batch_motion = outs_planning[0][layer_id]['agent_fut_preds']
                batch_motion_cls = outs_planning[0][layer_id]['agent_fut_preds_cls']
            else:
                batch_motion = None
                batch_motion_cls = None

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                batch_iou,
                batch_motion,
                batch_motion_cls,
                self.iou_rescore_weight,
                filter=True
            )

            if self.test_cfg["dataset"] == "nuScenes":
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=["pedestrian"],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=["traffic_cone"],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg["dataset"] == "Waymo":
                self.tasks = [
                    dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                    dict(
                        num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7
                    ),
                    dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]["bboxes"]
                scores = temp[i]["scores"]
                labels = temp[i]["labels"]

                keep_mask = torch.zeros_like(scores)
                for task in self.tasks:
                    task_mask = torch.zeros_like(scores)
                    for cls_idx in task["indices"]:
                        task_mask += labels == cls_idx
                    task_mask = task_mask.bool()
                    if self.test_cfg["nms_type"] is not None and task["radius"] > 0:
                        top_scores = scores[task_mask]
                        centers = boxes3d[task_mask][:, [0, 1]]
                        boxes_for_nms = torch.cat([centers, top_scores.view(-1, 1)], dim=1)

                        task_keep_indices = torch.tensor(
                            circle_nms(
                                boxes_for_nms.cpu().numpy(),
                                task["radius"],
                                post_max_size=self.test_cfg['post_max_size']))
                    else:
                        task_keep_indices = torch.arange(task_mask.sum())

                    if task_keep_indices.shape[0] != 0:
                        keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                        keep_mask[keep_indices] = 1
                keep_mask = keep_mask.bool()
                if with_motion:
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask],
                            trajs=temp[i]["trajs"][keep_mask], trajs_cls=temp[i]["trajs_cls"][keep_mask])
                            #    trajs=temp[i]["trajs"][keep_mask])
                            
                else:
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [
            [
                LiDARInstance3DBoxes(
                    rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1]
                ),
                rets[0][0]["scores"],
                rets[0][0]["labels"].int(),
            ]
        ]

        if with_motion:
            res[0].append(rets[0][0]["trajs"])
            res[0].append(rets[0][0]["trajs_cls"])
        return res
