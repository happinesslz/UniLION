import torch
import torch.nn as nn
import torch_scatter
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models.builder import NECKS
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp.autocast_mode import autocast

from projects.mmdet3d_plugin.models.ops import bev_pool_v2


@NECKS.register_module(force=True)
class LSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
            self,
            grid_config,
            input_size,
            k=4,
            downsample=16,
            in_channels=512,
            out_channels=64,
            bev_out_channels=128,
            accelerate=False,
            sid=False,
            out_voxel=True,
            out_bev=True,
            with_depth_w=False,
            with_cp=False,
            with_depth_from_lidar=False,
    ):
        super(LSSViewTransformer, self).__init__()
        self.with_cp = with_cp
        self.grid_config = grid_config
        self.k = k
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.bev_out_channels = bev_out_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.out_voxel = out_voxel
        self.out_bev = out_bev
        self.with_depth_w = with_depth_w
        self.with_depth_from_lidar = with_depth_from_lidar

        if self.with_depth_from_lidar:
            self.lidar_input_net = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=int(2 * self.downsample / 8),
                          padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True))
            out_channels = self.D + self.out_channels
            self.depth_net = nn.Sequential(
                nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, 1))

        if self.out_bev:
            self.bev_out = nn.Conv2d(int(self.grid_size[2]) * self.out_channels, bev_out_channels, kernel_size=1)
        if self.out_voxel:
            self.out_norm = nn.LayerNorm(self.out_channels)

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 4, 4).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2], ], dim=-1)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float) \
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D - 1) *
                              torch.log((depth_cfg_t[1] - 1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat, dtype=torch.float) \
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat, dtype=torch.float) \
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        return torch.stack((x, y, d), -1)

    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)
        points = bda[:, :3, :3].view(B, 1, 1, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)).squeeze(-1)
        points += bda[:, :3, 3].view(B, 1, 1, 1, 1, 3)
        return points

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        assert ranks_feat is not None

        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)

        voxel_feat = None
        voxel_indices = None
        bev_feat = None

        if self.out_bev:
            bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                                   bev_feat_shape, interval_starts,
                                   interval_lengths)
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)

        if self.out_voxel:
            B, N, _, _, _ = depth.shape

            # img_d = (depth * self.frustum[None, None, :, :, :, 2].to(depth)).sum(dim=2).unsqueeze(-1)
            # img_xy = self.frustum[None, None, 0, :, :, 0:2].to(depth).repeat(B, N, 1, 1, 1)
            # img_coords = torch.cat([img_xy, img_d], axis=-1)
            #
            # points = img_coords - post_trans.view(B, N, 1, 1, 3)
            # points = torch.inverse(post_rots).view(B, N, 1, 1, 3, 3) \
            #     .matmul(points.unsqueeze(-1))
            #
            # # cam_to_ego
            # points = torch.cat(
            #     (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 4)
            # combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
            # points = combine.view(B, N, 1, 1, 3, 3).matmul(points).squeeze(-1)
            # points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 3)
            # points = bda[:, :3, :3].view(B, 1, 1, 1, 3, 3).matmul(
            #     points.unsqueeze(-1)).squeeze(-1)
            # points += bda[:, :3, 3].view(B, 1, 1, 1, 3)
            #
            # points = ((points - self.grid_lower_bound.view(1, 1, 1, 1, 3).to(points)) /
            #           self.grid_interval.view(1, 1, 1, 1, 3).to(points))
            #
            # voxel_indices = list()
            # voxel_feats = list()
            # for b in range(B):
            #     voxel_pos = points[b].reshape(-1, 3)
            #     mask = ((voxel_pos[:, 0] >= 0) & (voxel_pos[:, 0] < self.grid_size[0]) &
            #             (voxel_pos[:, 1] >= 0) & (voxel_pos[:, 1] < self.grid_size[1]) &
            #             (voxel_pos[:, 2] >= 0) & (voxel_pos[:, 2] < self.grid_size[2]))
            #
            #     voxel_pos = voxel_pos[mask]
            #     voxel_feat = feat[b].reshape(-1, feat.shape[-1])[mask]
            #     voxel_pos = torch.cat([torch.zeros_like(voxel_pos[..., 0:1]) + b, voxel_pos.flip(-1)], dim=-1).int()
            #     voxel_indices.append(voxel_pos)
            #     voxel_feats.append(voxel_feat)
            #
            # voxel_indices = torch.cat(voxel_indices, dim=0)
            # voxel_feat = torch.cat(voxel_feats, dim=0)

            ranks_bev, ranks_depth, ranks_feat, ranks_depth_w = self.voxel_pooling_prepare_v3(coor, depth, self.k)
            voxel_feat = feat.reshape(-1, feat.shape[-1])[ranks_feat]
            unq_coords, unq_inv = torch.unique(ranks_bev, return_inverse=True, return_counts=False, dim=0)
            if self.with_depth_w:
                voxel_feat = torch_scatter.scatter_add(voxel_feat * ranks_depth_w.unsqueeze(-1), unq_inv, dim=0)
            else:
                voxel_feat = torch_scatter.scatter_add(voxel_feat, unq_inv, dim=0)

            scale_xyz = (self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
            scale_xy = (self.grid_size[1] * self.grid_size[0])
            scale_x = (self.grid_size[0])

            unq_coords = unq_coords.int()
            voxel_batch = unq_coords // scale_xyz
            voxel_indices = unq_coords % scale_xyz
            voxel_z = voxel_indices // scale_xy
            voxel_y = (voxel_indices % scale_xy) // scale_x
            voxel_x = voxel_indices % scale_x
            voxel_indices = torch.stack([voxel_batch, voxel_z, voxel_y, voxel_x]).permute(1, 0)

        return voxel_feat, voxel_indices, bev_feat

    def voxel_pooling_prepare_v3(self, coor, depth, k=4):
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W

        select = depth.topk(k=k, dim=2).indices.permute(0, 1, 3, 4, 2).reshape(-1, k)
        index = torch.ones_like(depth).nonzero().reshape(-1, 5)
        index = index[:, 0] * (N * D * H * W) + index[:, 1] * (D * H * W) + index[:, 2] * (H * W) + index[:,
                                                                                                    3] * W + index[:, 4]
        index = index.reshape(B, N, D, H, W).permute(0, 1, 3, 4, 2).reshape(-1, D)

        index = torch.gather(index, dim=1, index=select).reshape(-1)

        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        coor = coor[index]
        ranks_depth = ranks_depth[index]
        ranks_feat = ranks_feat[index]
        ranks_depth_w = depth.flatten()[index]
        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])

        if len(kept) == 0:
            return None, None, None, None
        coor, ranks_depth, ranks_feat, ranks_depth_w = \
            coor[kept], ranks_depth[kept], ranks_feat[kept], ranks_depth_w[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
                self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat, ranks_depth_w = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order], ranks_depth_w[order]

        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), ranks_depth_w.contiguous()

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])

        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
                self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        coor = self.get_lidar_coor(*input[1:7])
        voxel_feat, voxel_indices, bev_feat = self.voxel_pooling_v2(
            coor, depth.view(B, N, self.D, H, W),
            tran_feat.view(B, N, self.out_channels, H, W), *input[1:7])

        if self.out_voxel:
            voxel_feat = self.out_norm(voxel_feat)
            voxel_indices = voxel_indices.int()
        if self.out_bev:
            bev_feat = self.bev_out(bev_feat)
        return voxel_feat, voxel_indices, bev_feat, depth

    def view_transform(self, input, depth, tran_feat):
        for shape_id in range(3):
            assert depth.shape[shape_id + 1] == self.frustum.shape[shape_id]
        return self.view_transform_core(input, depth, tran_feat)

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                    1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return depth_loss

    def forward(self, input, depth_from_lidar=None):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        (x, sensor2keyegos, ego2globals, intrins,
         post_rots, post_trans, bda, mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_depth_from_lidar:
            assert depth_from_lidar is not None
            if isinstance(depth_from_lidar, list):
                assert len(depth_from_lidar) == 1
                depth_from_lidar = depth_from_lidar[0]
            h_img, w_img = depth_from_lidar.shape[2:]
            depth_from_lidar = depth_from_lidar.view(B * N, 1, h_img, w_img)
            depth_from_lidar = self.lidar_input_net(depth_from_lidar)
            x = torch.cat([x, depth_from_lidar], dim=1)

        if self.with_cp:
            x = checkpoint(self.depth_net, x)
        else:
            x = self.depth_net(x)

        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)