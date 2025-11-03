import copy


import torch
import torch.nn as nn
import torch_scatter

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import (
    DETECTORS,
    BaseDetector
)
from mmdet3d.models import builder
import spconv.pytorch as spconv
from einops import rearrange

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from .utils.merge_augs import merge_aug_bboxes_3d
from .utils.util import freeze_module
from ..core.bbox.transforms import bbox3d2result, map3d2result, occ3d2result, planning3d2result, seg_map3d2result
from projects.mmdet3d_plugin.models.utils import modulate


@DETECTORS.register_module()
class UniLION(BaseDetector):
    def __init__(
            self,
            pts_voxel_encoder=None,
            pts_backbone=None,
            tasks=None,
            seq=False,
            max_seq_time=2,
            fusion=False,
            with_depth_loss=False,
            loss_depth_weight=1.0,
            voxel_size=None,
            pc_range=None,
            img_backbone=None,
            bev_backbone=None,
            map2bev=None,
            pts_neck=None,
            img_neck=None,
            img_map2bev=None,
            img_bev_backbone=None,
            img_bev_neck=None,
            fusion_layer=None,
            proj_layer=None,
            det_head=None,
            occ_head=None,
            map_head=None,
            motion_head=None,
            planning_head=None,
            init_cfg=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            use_grid_mask=False,
    ):
        super(UniLION, self).__init__(init_cfg=init_cfg)

        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_backbone = pts_backbone
        self.img_backbone = img_backbone
        self.bev_backbone = bev_backbone
        self.map2bev = map2bev
        self.pts_neck = pts_neck
        self.img_neck = img_neck
        self.img_map2bev = img_map2bev
        self.img_bev_backbone = img_bev_backbone
        self.img_bev_neck = img_bev_neck
        self.proj_layer = proj_layer
        self.fusion_layer = fusion_layer
        self.det_head = det_head
        self.occ_head = occ_head
        self.map_head = map_head
        self.motion_head = motion_head
        self.planning_head = planning_head

        self.tasks = tasks
        self.seq = seq
        self.fusion = fusion
        self.with_depth_loss = with_depth_loss
        self.loss_depth_weight = loss_depth_weight
        self.max_seq_time = max_seq_time
        self.use_grid_mask = use_grid_mask
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.train_cfg = train_cfg

        if self.with_motion:
            assert self.with_det
        if self.with_planning:
            assert self.with_det

        self._init_layers()

        if self.train_cfg is not None:
            self.freeze(self.train_cfg.get('freeze', None))

        self.reset()

    def _init_layers(self):
        if self.pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(self.pts_voxel_encoder)
        if self.pts_backbone:
            self.pts_backbone = builder.build_backbone(self.pts_backbone)
        if self.map2bev:
            self.map2bev = builder.build_middle_encoder(self.map2bev)
        self.bev_backbone = builder.build_backbone(self.bev_backbone)
        self.pts_neck = builder.build_neck(self.pts_neck)

        if self.img_backbone:
            self.img_backbone = builder.build_backbone(self.img_backbone)
        if self.img_neck:
            self.img_neck = builder.build_neck(self.img_neck)
        if self.img_map2bev:
            self.img_map2bev = builder.build_middle_encoder(self.img_map2bev)
        if self.img_bev_backbone:
            self.img_bev_backbone = builder.build_backbone(self.img_bev_backbone)
        if self.img_bev_neck:
            self.img_bev_neck = builder.build_neck(self.img_bev_neck)
        if self.proj_layer:
            self.proj_layer = builder.build_neck(self.proj_layer)
        if self.fusion_layer:
            self.fusion_layer = builder.build_neck(self.fusion_layer)
        if self.det_head and self.with_det:
            self.det_head = builder.build_head(self.det_head)
        if self.occ_head and self.with_occ:
            self.occ_head = builder.build_head(self.occ_head)
        if self.map_head and self.with_map:
            self.map_head = builder.build_head(self.map_head)
        if self.motion_head and self.with_motion:
            self.motion_head = builder.build_head(self.motion_head)
        if self.planning_head and self.with_planning:
            self.planning_head = builder.build_head(self.planning_head)

        if self.seq:
            hidden_size = self.pts_neck.in_channels[0]

            self.temporal_fuser = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False)
            )
                
        if self.use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
        else:
            self.grid_mask = None

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        return hasattr(self, 'pts_backbone') and self.img_backbone is not None

    @property
    def with_det(self):
        return self.tasks['det']

    @property
    def with_occ(self):
        return self.tasks['occ']

    @property
    def with_map(self):
        return self.tasks['map']

    @property
    def with_motion(self):
        return self.tasks['motion']

    @property
    def with_planning(self):
        return self.tasks['planning']

    @property
    def with_img(self):
        return self.img_backbone is not None

    @property
    def with_pts(self):
        return self.pts_voxel_encoder is not None

    def freeze(self, freeze_dict):
        if freeze_dict is None:
            return
        for name in freeze_dict:
            if hasattr(self, name) and getattr(self, name) is not None:
                freeze_module(getattr(self, name))

    def reset(self):
        self.prev_token = list()
        self.prev_features = list()
        self.prev_voxel_coords = list()
        self.prev_timestamps = list()

    def update(self, i, token, timestamps, features, voxel_coords):
        self.prev_token[i] = token
        self.prev_features[i] = features.detach()
        self.prev_voxel_coords[i] = voxel_coords
        if timestamps is not None:
            self.prev_timestamps[i] = timestamps

    def get(self, i):
        if len(self.prev_token) <= i:
            self.prev_token.append(None)
            self.prev_features.append(None)
            self.prev_voxel_coords.append(None)
            self.prev_timestamps.append(None)
        return self.prev_token[i], self.prev_timestamps[i], self.prev_features[i], self.prev_voxel_coords[i]

    def trans(self, voxel_features, voxel_coords, lidar2global, inverse=False):
        voxel_coords = voxel_coords.float()
        lidar2global = torch.tensor(lidar2global).to(voxel_coords)

        if inverse:
            voxel_coords[:, 1:4] = (torch.cat([voxel_coords[:, [3, 2, 1]], torch.ones_like(voxel_coords[:, 0:1])],
                                              dim=-1) @ torch.inverse(lidar2global).T)[..., [2, 1, 0]]

            voxel_coords[:, 3] = (voxel_coords[:, 3] - self.pc_range[0]) / self.voxel_size[0]
            voxel_coords[:, 2] = (voxel_coords[:, 2] - self.pc_range[1]) / self.voxel_size[1]
            voxel_coords[:, 1] = (voxel_coords[:, 1] - self.pc_range[2]) / self.voxel_size[2]
            voxel_coords = voxel_coords.int()
            grid_size = self.pts_backbone.sparse_shape
            grid_size = [grid_size[0], grid_size[1], grid_size[2]]
            z = voxel_coords[..., 1]
            y = voxel_coords[..., 2]
            x = voxel_coords[..., 3]
            mask = (x >= 0) & (x < grid_size[2]) & (y >= 0) & (y < grid_size[1]) & (z >= 0) & (z < grid_size[0])
            voxel_coords = voxel_coords[mask]
            voxel_features = voxel_features[mask]

            return voxel_features, voxel_coords
        else:
            voxel_coords[:, 3] = voxel_coords[:, 3] * self.voxel_size[0] + self.pc_range[0]
            voxel_coords[:, 2] = voxel_coords[:, 2] * self.voxel_size[1] + self.pc_range[1]
            voxel_coords[:, 1] = voxel_coords[:, 1] * self.voxel_size[2] + self.pc_range[2]

            voxel_coords[:, 1:4] = (torch.cat([voxel_coords[:, [3, 2, 1]], torch.ones_like(voxel_coords[:, 0:1])],
                                              dim=-1) @ lidar2global.T)[..., [2, 1, 0]]
            return voxel_features, voxel_coords

    def forward_temporal(self, voxels, **data):
        start_flags = list()
        prev_voxel_features_list = list()
        prev_voxel_coords_list = list()

        voxel_features = voxels.features
        voxel_coords = voxels.indices
        cur_voxel_features = voxel_features

        for i, meta in enumerate(data['img_metas']):
            scene_token = meta['scene_token']
            timestamp = meta['timestamp']
            prev_token, prev_timestamp, prev_features, prev_voxel_coords = self.get(i)

            if prev_token is None:
                is_start = True
            else:
                is_start = prev_token != scene_token
                is_start |= self.max_seq_time < (timestamp - prev_timestamp)
            start_flags.append(is_start)

            if not is_start:
                select = voxel_coords[:, 0] == i
                trans_features, trans_voxel_coords = self.trans(prev_features, prev_voxel_coords,
                                                                meta['lidar2global'], inverse=True)
                prev_voxel_features_list.append(trans_features)
                prev_voxel_coords_list.append(trans_voxel_coords)

        if len(prev_voxel_features_list) != 0:
            prev_voxel_features = torch.cat(prev_voxel_features_list, dim=0)
            prev_voxel_coords = torch.cat(prev_voxel_coords_list, dim=0)

            prev_voxels = spconv.SparseConvTensor(
                features=prev_voxel_features,
                indices=prev_voxel_coords.int(),
                spatial_shape=voxels.spatial_shape,
                batch_size=voxels.batch_size
            )

            prev_voxels_features = prev_voxel_features
        else:
            prev_voxels_features = torch.zeros_like(cur_voxel_features[0:1]).detach()
            prev_voxel_coords = torch.zeros_like(voxel_coords[0:1])
    
        prev_voxel_features = self.temporal_fuser(prev_voxels_features)
        temporal_features = torch.cat([prev_voxel_features, cur_voxel_features], axis=0)
        temporal_indices = torch.cat([prev_voxel_coords, voxel_coords], axis=0)

        x = spconv.SparseConvTensor(
            features=temporal_features,
            indices=temporal_indices.int(),
            spatial_shape=voxels.spatial_shape,
            batch_size=voxels.batch_size
        )

        return start_flags, x

    def store_temporal(self, x, start_flags, **data):
        cur_x = x.features
        cur_x_indices = x.indices

        for i in range(x.batch_size):
            select = cur_x_indices[:, 0] == i
            scene_token = data['img_metas'][i]['scene_token']
            if start_flags[i]:
                timestamp = data['img_metas'][i]['timestamp']
            else:
                timestamp = None

            trans_features, trans_indices = self.trans(cur_x[select], cur_x_indices[select],
                                                       data['img_metas'][i]['lidar2global'])
            self.update(i, scene_token, timestamp, trans_features, trans_indices)

    def prepare_inputs(self, img, inputs):
        # split the inputs into each frame
        assert len(inputs) == 6
        B, N, C, H, W = img.shape
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from adj sensor to key ego
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)  # (B, 1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())  # (B, 1, 4, 4)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()  # (B, N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()

        return [sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def image_encoder(self, img):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.grid_mask is not None:
            img = self.grid_mask(img)
        feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        return feature_maps

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_img_feat(self, img, **data):
        if not self.with_img_backbone:
            return None, None, None, None, None, None
        feature_maps = self.image_encoder(img)
        img_inputs = data['img_inputs']
        img_inputs = self.prepare_inputs(img, img_inputs)
        if self.img_map2bev is not None:
            mlp_input = [self.img_map2bev.get_mlp_input(*img_inputs)]
            voxel_feat, voxel_indices, bev_feat, depth = self.img_map2bev(feature_maps + img_inputs + mlp_input,
                                                                          depth_from_lidar=data['gt_depth'])
        else:
            voxel_feat, voxel_indices, bev_feat, depth = None, None, None, None
        return voxel_feat, voxel_indices, bev_feat, feature_maps, img_inputs, depth

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, voxel_features, voxel_coords, img_bev_feat, **data):
        """Extract features of points."""

        # if self.pts_backbone:
        if self.pts_backbone and (voxel_features is not None):
            batch_size = int(voxel_coords[-1, 0]) + 1

            if self.seq:
                x = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_coords.int(),
                    spatial_shape=self.pts_backbone.sparse_shape,
                    batch_size=batch_size
                )
                start_flags, new_x = self.forward_temporal(x, **data)
                x = new_x
                self.store_temporal(new_x, start_flags, **data)
                x = self.pts_backbone(x.features, x.indices, batch_size)
            else:
                x = self.pts_backbone(voxel_features, voxel_coords, batch_size)

            if self.map2bev:
                x = self.map2bev(x)
        else:
            x = img_bev_feat

        if self.fusion_layer is not None:
            if img_bev_feat is None:
                img_bev_feat = torch.zeros_like(x)
            x = self.fusion_layer(x, img_bev_feat)

        x = self.bev_backbone(x)
        x = self.pts_neck(x)
        return x

    def forward(self, points, img, **data):
        if self.training:
            return self.forward_train(points, img, **data)
        else:
            return self.forward_test(points, img, **data)

    def extract_feat(self, points, img, **data):
        if self.with_pts:
            pts_voxel_features, pts_voxel_coords = self.pts_voxel_encoder(points)

        img_bev_feat = None
        depth = None

        if self.with_img:
            (img_voxel_features, img_voxel_coords,
             img_bev_feat, feature_maps, img_inputs, depth) = self.extract_img_feat(img, **data)

        if self.fusion:
            assert self.with_img and self.with_pts

            voxel_features = pts_voxel_features
            voxel_coords = pts_voxel_coords

            if img_voxel_features is not None:
                voxel_features = torch.cat([voxel_features, img_voxel_features], dim=0)
                voxel_coords = torch.cat([voxel_coords, img_voxel_coords], dim=0)
        elif self.with_pts:
            voxel_features = pts_voxel_features
            voxel_coords = pts_voxel_coords
        elif self.with_img:
            voxel_features = img_voxel_features
            voxel_coords = img_voxel_coords

        return self.extract_pts_feat(voxel_features, voxel_coords, img_bev_feat, **data), depth

    def forward_train(self, points, img, **data):
        metas = data['img_metas']

        tasks = [meta['task'] for meta in metas]
        for t in tasks:
            assert t == tasks[0]
        task = tasks[0]

        dynamic = self.train_cfg.get('dynamic', False)
        loss_det_weight = self.train_cfg.get('loss_det_weight', 1)
        loss_occ_weight = self.train_cfg.get('loss_occ_weight', 1)
        loss_map_weight = self.train_cfg.get('loss_map_weight', 1)
        loss_motion_weight = self.train_cfg.get('loss_motion_weight', 1)
        loss_planning_weight = self.train_cfg.get('loss_planning_weight', 1)

        pts_feats, depth = self.extract_feat(points, img, **data)

        if self.with_det:
            outs_det = self.det_head(pts_feats)

        if self.with_occ:
            outs_occ = self.occ_head(pts_feats)

        if self.with_map:
            outs_map = self.map_head(pts_feats)

        if self.with_motion:
            det_outs = outs_det[0]
            outs_motion = self.motion_head(pts_feats, pts_feats, det_outs)

        if self.with_planning:
            outs_planning = self.planning_head(pts_feats, pts_feats, data['ego_status'], data['gt_ego_fut_cmd'], data['gt_ego_fut_trajs'], data['img_metas'])

        losses = dict()

        if self.with_depth_loss and depth is not None:
            gt_depth = data['gt_depth']
            loss_depth = self.img_map2bev.get_depth_loss(gt_depth, depth)
            losses.update({'loss_depth': loss_depth * self.loss_depth_weight})

        if self.with_det:
            loss_det = 0
            losses_det, assigned_gt_inds = \
                self.det_head.loss(outs_det, metas, data['gt_bboxes_3d'], data['gt_labels_3d'])
            for key in losses_det.keys():
                loss_det += losses_det[key]

            if 'detection' not in task:
                loss_det_weight = 0

            for key in losses_det.keys():
                losses_det[key] = losses_det[key] * loss_det_weight

            losses.update(losses_det)

        if self.with_occ:
            loss_occ = 0
            losses_occ = self.occ_head.loss(outs_occ, metas, data['gt_occ'], data['mask_camera'], data['gt_occ_flow'] if 'gt_occ_flow' in data else None)
            for key in losses_occ.keys():
                loss_occ += losses_occ[key]

            if 'occ' not in task:
                loss_occ_weight = 0
            if dynamic:
                loss_occ_weight = (loss_det / (loss_occ + 1e-5)).detach() * loss_occ_weight

            for key in losses_occ.keys():
                losses_occ[key] = losses_occ[key] * loss_occ_weight

            losses.update(losses_occ)

        if self.with_map:
            loss_map = 0
            losses_map = self.map_head.loss(outs_map, metas, data['gt_bev_masks'])
            for key in losses_map.keys():
                loss_map += losses_map[key]

            if 'map' not in task:
                loss_map_weight = 0
            if dynamic:
                loss_map_weight = (loss_det / (loss_map + 1e-5)).detach() * loss_map_weight

            for key in losses_map.keys():
                losses_map[key] = losses_map[key] * loss_map_weight

            losses.update(losses_map)

        if self.with_motion:
            loss_motion = 0
            losses_motion = self.motion_head.loss(outs_motion, metas, assigned_gt_inds,
                                                  data['gt_agent_fut_trajs'], data['gt_agent_fut_masks'])

            for key in losses_motion.keys():
                loss_motion += losses_motion[key]

            if dynamic:
                loss_motion_weight = (loss_det / (loss_motion + 1e-5)).detach() * loss_motion_weight

            for key in losses_motion.keys():
                losses_motion[key] = losses_motion[key] * loss_motion_weight

            losses.update(losses_motion)

        if self.with_planning:
            loss_planning = 0
            losses_planning = self.planning_head.loss(outs_planning, metas,
                                                      data['gt_ego_fut_trajs'], data['gt_ego_fut_masks'],
                                                      data['gt_ego_fut_cmd'], data['ego_status'])

            for key in losses_planning.keys():
                loss_planning += losses_planning[key]

            if dynamic:
                loss_planning_weight = (loss_det / (loss_planning + 1e-5)).detach() * loss_planning_weight

            for key in losses_planning.keys():
                losses_planning[key] = losses_planning[key] * loss_planning_weight

            losses.update(losses_planning)

        return losses

    def forward_test(self, points, img, **data):
        if isinstance(img, list):
            return self.aug_test(points, img, **data)
        else:
            return self.simple_test(points, img, **data)

    def simple_test(self, points, img, **data):
        pts_feats, depth = self.extract_feat(points, img, **data)

        metas = data['img_metas']

        res_list = [dict() for i in range(len(metas))]

        if self.with_det:
            outs_det = self.det_head(pts_feats)

        if self.with_occ:
            outs_occ = self.occ_head(pts_feats)
            results_occ = self.occ_head.get_bboxes(outs_occ, metas, **data)
            occ_results = [
                occ3d2result(occ)
                for occ in results_occ
            ]
            for result_dict, pts_occ in zip(res_list, occ_results):
                result_dict['pts_occ'] = pts_occ

        if self.with_map:
            outs_map = self.map_head(pts_feats)
            results_seg, results_map = self.map_head.get_bboxes(outs_map, metas, **data)

            seg_results = [
                seg_map3d2result(seg)
                for seg in results_seg
            ]
            for result_dict, pts_seg_map in zip(res_list, seg_results):
                result_dict['pts_seg_map'] = pts_seg_map

            if results_map is not None:
                map_results = [
                    map3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels in results_map
                ]
                for result_dict, pts_map in zip(res_list, map_results):
                    result_dict['pts_map'] = pts_map

        if self.with_motion:
            det_outs = outs_det[0]
            outs_motion = self.motion_head(pts_feats, pts_feats, det_outs)

        if self.with_planning:
            outs_planning = self.planning_head(pts_feats, pts_feats, data['ego_status'], data['gt_ego_fut_cmd'], None, data['img_metas'])
            results_planning = self.planning_head.get_bboxes(outs_planning, metas, **data)
            planning_results = [
                planning3d2result(pts_plan_reg)
                for pts_plan_reg in results_planning
            ]
            for result_dict, pts_plan_reg in zip(res_list, planning_results):
                result_dict['pts_plan_reg'] = pts_plan_reg

        if self.with_det:
            if self.with_motion:
                results_det = self.det_head.get_bboxes(outs_det, outs_motion, metas, **data)
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels, trajs, trajs_cls)
                    for bboxes, scores, labels, trajs, trajs_cls in results_det
                ]
            else:
                results_det = self.det_head.get_bboxes(outs_det, None, metas, **data)
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels in results_det
                ]

            for result_dict, pts_bbox in zip(res_list, bbox_results):
                result_dict['pts_bbox'] = pts_bbox

        return res_list

    def aug_test(self, points, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(points[0], img[0], **data)

    # def aug_test(self, points, img, **data):
    #     aug_bboxes = list()
    #     data_list = [dict() for _ in range(len(points))]

    #     for key in data.keys():
    #         for i in range(len(data_list)):
    #             if isinstance(data[key], list):
    #                 data_list[i][key] = data[key][i]
    #             else:
    #                 data_list[i][key] = data[key]

    #     for single_points, single_img, single_data in zip(points, img, data_list):
    #         outs = self.simple_test(single_points, single_img, **single_data)
    #         aug_bboxes.append(outs[0]['pts_bbox'])

    #     if len(aug_bboxes) == 1:
    #         return outs
    #     merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, data_list, self.det_head.test_cfg)
    #     res_list = [dict()]
    #     res_list[0]['pts_bbox'] = merged_bboxes
    #     return res_list