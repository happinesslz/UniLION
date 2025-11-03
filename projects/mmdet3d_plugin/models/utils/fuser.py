import torch
import torch.nn as nn
import numpy as np
from mmdet3d.models.builder import NECKS


@NECKS.register_module(force=True)
class ConvFuser(nn.Module):
    def __init__(self, img_in_channels, out_channels=256):
        super(ConvFuser, self).__init__()
        self.fuser = nn.Sequential(
            nn.Conv2d(img_in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x1, x2):
        assert x2.size() == x2.size()
        # print('###x1:', x1.shape)
        # print('###x2:', x2.shape)
        x = torch.cat([x1, x2], dim=1)
        y = self.fuser(x)
        return y


@NECKS.register_module(force=True)
class ProjFuser(nn.Module):
    def __init__(self, img_in_channels, out_channels, voxel_size, pc_range, downsample, grid_config):
        super(ProjFuser, self).__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.downsample = downsample
        self.grid_config = grid_config

        self.compressed_img_feats = nn.Sequential(
            nn.Linear(256, img_in_channels, bias=False),
            # nn.BatchNorm1d(img_in_channels, eps=1e-3, momentum=0.01),
            # nn.ReLU(True)
        )
        self.fuser = nn.Sequential(
            nn.Linear(img_in_channels + out_channels, out_channels, bias=False),
            # nn.LayerNorm(out_channels),
            # nn.GELU()
        )

    def gather_img_feature(self, points, img_feature, img):
        height, width = img_feature.shape[1], img_feature.shape[2]
        # height, width = img.shape[1], img.shape[2]
        # self.downsample = 1

        coor = torch.round(points[:, :2] / self.downsample)  # (N_points, 2)  2: (u, v)
        depth = points[:, 2]  # (N_points, )
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                        depth < self.grid_config['depth'][1]) & (
                        depth >= self.grid_config['depth'][0])
        gather_feature = torch.zeros([len(points), img_feature.shape[0]]).to(img_feature)
        coor, depth = coor[kept1], depth[kept1]
        coor = coor.int()
        img_feature = img_feature[:, coor[:, 1], coor[:, 0]]
        gather_feature[kept1] = img_feature.T

        # import matplotlib.pyplot as plt
        # img = img.permute(1, 2, 0)
        # mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        # std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        # img = img.cpu().numpy()
        # img = (img * std + mean).astype(np.uint8)
        # plt.figure(figsize=(width/100, height/100), dpi=100)  # 设置精确尺寸
        # plt.imshow(img)
        # plt.scatter(coor.cpu().numpy()[:, 0], coor.cpu().numpy()[:, 1], c='red', s=5, alpha=0.7, edgecolors='none')  # 红色半透明散点
        #
        # # 保存图像（去除边框和空白）
        # plt.axis('off')
        # plt.savefig('/data/jhhou/disk/temp/image_with_scatter.png',
        #         bbox_inches='tight',
        #         pad_inches=0,
        #         dpi=100)
        # plt.close()
        # import pdb;pdb.set_trace()
        return gather_feature

    def forward(self, voxel_features, voxel_coords, img_features, img_inputs, img_metas, imgs):
        bs = len(img_metas)
        fused_features = list()
        fused_voxel_coords = list()

        for b in range(bs):
            points_camego_aug = voxel_coords[voxel_coords[:, 0] == b][:, [3, 2, 1]].float()
            voxel_feature = voxel_features[voxel_coords[:, 0] == b]
            fused_voxel_coords.append(voxel_coords[voxel_coords[:, 0] == b])

            points_camego_aug[:, 0] = points_camego_aug[:, 0] * self.voxel_size[0] + self.pc_range[0]
            points_camego_aug[:, 1] = points_camego_aug[:, 1] * self.voxel_size[1] + self.pc_range[1]
            points_camego_aug[:, 2] = points_camego_aug[:, 2] * self.voxel_size[2] + self.pc_range[2]

            rots, trans, intrins = img_inputs[:3]
            rots = rots[b]
            trans = trans[b]
            intrins = intrins[b]
            post_rots, post_trans, bda = img_inputs[3:]
            post_rots = post_rots[b]
            post_trans = post_trans[b]
            bda = bda[b]

            points_camego = points_camego_aug - bda[:3, 3].view(1, 3)
            points_camego = points_camego.matmul(torch.inverse(bda[:3, :3]).T)
            lidar2cams = img_metas[b]['lidar2cam']

            img_feats_list = []

            for cid in range(len(lidar2cams)):
                lidar2cam = lidar2cams[cid]
                lidar2cam = torch.tensor(lidar2cam).to(points_camego)

                cam2img = np.eye(4, dtype=np.float32)
                cam2img = torch.from_numpy(cam2img).to(points_camego)
                cam2img[:3, :3] = intrins[cid]
                lidar2img = cam2img.matmul(lidar2cam.T)

                points_img = points_camego.matmul(
                    lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                points_img = torch.cat(
                    [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                    1)
                points_img = points_img.matmul(
                    post_rots[cid].T) + post_trans[cid:cid + 1, :]

                img_feature = img_features[0][b][cid]

                img = imgs[b][cid]

                img_feature = self.gather_img_feature(points_img, img_feature, img)
                img_feats_list.append(img_feature.unsqueeze(0))

            img_feature = torch.cat(img_feats_list, dim=0).sum(dim=0)
            img_feature = self.compressed_img_feats(img_feature)
            fusion_feature = torch.cat([voxel_feature, img_feature], dim=1)
            # fusion_feature = torch.cat([voxel_feature, 0.0*img_feature], dim=1)
            # print('img features as 0')
            voxel_feature = self.fuser(fusion_feature)
            fused_features.append(voxel_feature)
        fused_features = torch.cat(fused_features, dim=0)
        fused_voxel_coords = torch.cat(fused_voxel_coords, dim=0)
        return fused_features, fused_voxel_coords


@NECKS.register_module(force=True)
class ProjViewTransformer(nn.Module):
    def __init__(self, img_in_channels, out_channels, voxel_size, pc_range, downsample, grid_config, fusion_manner='concat'):
        super(ProjViewTransformer, self).__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.downsample = downsample
        self.grid_config = grid_config

        self.compressed_img_feats = nn.Sequential(
            nn.Linear(256, img_in_channels, bias=False),
        )
        self.fusion_manner = fusion_manner

    def gather_img_feature(self, points, img_feature):
        height, width = img_feature.shape[1], img_feature.shape[2]

        coor = torch.round(points[:, :2] / self.downsample)  # (N_points, 2)  2: (u, v)
        depth = points[:, 2]  # (N_points, )
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                        depth < self.grid_config['depth'][1]) & (
                        depth >= self.grid_config['depth'][0])
        gather_feature = torch.zeros([len(points), img_feature.shape[0]]).to(img_feature)
        coor, depth = coor[kept1], depth[kept1]
        coor = coor.int()
        img_feature = img_feature[:, coor[:, 1], coor[:, 0]]
        gather_feature[kept1] = img_feature.T

        return gather_feature

    def forward(self, voxel_features, voxel_coords, img_features, img_inputs, img_metas, imgs):
        bs = len(img_metas)
        img_voxel_features = list()
        img_voxel_coords = list()
        pts_voxel_features = list()
        pts_voxel_coords = list()

        for b in range(bs):
            points_camego_aug = voxel_coords[voxel_coords[:, 0] == b][:, [3, 2, 1]].float()
            voxel_feature = voxel_features[voxel_coords[:, 0] == b]
            img_voxel_coords.append(voxel_coords[voxel_coords[:, 0] == b])
            pts_voxel_features.append(voxel_feature)
            pts_voxel_coords.append(voxel_coords[voxel_coords[:, 0] == b])

            points_camego_aug[:, 0] = points_camego_aug[:, 0] * self.voxel_size[0] + self.pc_range[0]
            points_camego_aug[:, 1] = points_camego_aug[:, 1] * self.voxel_size[1] + self.pc_range[1]
            points_camego_aug[:, 2] = points_camego_aug[:, 2] * self.voxel_size[2] + self.pc_range[2]

            rots, trans, intrins = img_inputs[:3]
            rots = rots[b]
            trans = trans[b]
            intrins = intrins[b]
            post_rots, post_trans, bda = img_inputs[3:]
            post_rots = post_rots[b]
            post_trans = post_trans[b]
            bda = bda[b]

            points_camego = points_camego_aug - bda[:3, 3].view(1, 3)
            points_camego = points_camego.matmul(torch.inverse(bda[:3, :3]).T)
            lidar2cams = img_metas[b]['lidar2cam']

            img_feats_list = []

            for cid in range(len(lidar2cams)):
                lidar2cam = lidar2cams[cid]
                lidar2cam = torch.tensor(lidar2cam).to(points_camego)

                cam2img = np.eye(4, dtype=np.float32)
                cam2img = torch.from_numpy(cam2img).to(points_camego)
                cam2img[:3, :3] = intrins[cid]
                lidar2img = cam2img.matmul(lidar2cam.T)

                points_img = points_camego.matmul(
                    lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
                points_img = torch.cat(
                    [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                    1)
                points_img = points_img.matmul(
                    post_rots[cid].T) + post_trans[cid:cid + 1, :]

                img_feature = img_features[0][b][cid]

                img_feature = self.gather_img_feature(points_img, img_feature)
                img_feats_list.append(img_feature.unsqueeze(0))
            img_feature = torch.cat(img_feats_list, dim=0).sum(dim=0)
            img_feature = self.compressed_img_feats(img_feature)
            img_voxel_features.append(img_feature)

        img_voxel_features = torch.cat(img_voxel_features, dim=0)
        img_voxel_coords = torch.cat(img_voxel_coords, dim=0)
        pts_voxel_features = torch.cat(pts_voxel_features, dim=0)
        pts_voxel_coords = torch.cat(pts_voxel_coords, dim=0)

        if self.fusion_manner=='add':
            voxel_features = pts_voxel_features + img_voxel_features
            voxel_coords = pts_voxel_coords
        else:
            voxel_features = torch.cat([pts_voxel_features, img_voxel_features], dim=0)
            voxel_coords = torch.cat([pts_voxel_coords, img_voxel_coords], dim=0)
        return voxel_features, voxel_coords
