import torch
import numpy as np
import mmcv
from mmcv.image.photometric import imnormalize
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets import LoadAnnotations3D
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import PIPELINES
from torchvision.transforms.functional import rotate


@PIPELINES.register_module()
class PrepareImageInputs(object):
    def __init__(
            self,
            data_config,
            is_train=False,
    ):
        self.is_train = is_train
        self.data_config = data_config

    def normalize_img(self, img):
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        to_rgb = True
        img = imnormalize(np.array(img), mean, std, to_rgb)
        img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
        return img

    def choose_cams(self):
        """
        Returns:
            cam_names: List[CAM_Name0, CAM_Name1, ...]
        """
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        """
        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resize比例float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: 随机旋转角度float
        """
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])    # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))            # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH     # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))       # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """
        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """
        Args:
            info:
            cam_name: 当前要读取的CAM.
        Returns:
            sensor2ego: (4, 4)
            ego2global: (4, 4)
        """
        w, x, y, z = info['cams'][cam_name]['sensor2ego_rotation']      # 四元数格式
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        sensor2ego_tran = torch.Tensor(
            info['cams'][cam_name]['sensor2ego_translation'])   # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = info['cams'][cam_name]['ego2global_rotation']      # 四元数格式
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)     # (3, 3)
        ego2global_tran = torch.Tensor(
            info['cams'][cam_name]['ego2global_translation'])   # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """
        Args:
            results:
            flip:
            scale:

        Returns:
            imgs:  (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins:     (N_views, 3, 3)
            post_rots:   (N_views, 3, 3)
            post_trans:  (N_views, 3)
        """
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []

        for cam_name in cam_names:
            cam_data = results['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # 初始化图像增广的旋转和平移矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # 当前相机内参
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # 获取当前相机的sensor2ego(4x4), ego2global(4x4)矩阵.
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results, cam_name)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs

            # img: PIL.Image;  post_rot: Tensor (2, 2);  post_tran: Tensor (2, )
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            # 以3x3矩阵表示图像的增广
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))    # 保存未归一化的图像，应该是为了做可视化.
            imgs.append(self.normalize_img(img))

            intrins.append(intrin)      # 相机内参 (3, 3)
            sensor2egos.append(sensor2ego)      # camera2ego变换 (4, 4)
            ego2globals.append(ego2global)      # ego2global变换 (4, 4)
            post_rots.append(post_rot)          # 图像增广旋转 (3, 3)
            post_trans.append(post_tran)        # 图像增广平移 (3, ）

        imgs = torch.stack(imgs)    # (N_views, 3, H, W)        # N_views = 6 * (N_history + 1)

        sensor2egos = torch.stack(sensor2egos)      # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)      # (N_views, 4, 4)
        intrins = torch.stack(intrins)              # (N_views, 3, 3)
        post_rots = torch.stack(post_rots)          # (N_views, 3, 3)
        post_trans = torch.stack(post_trans)        # (N_views, 3)

        lidar2img = results['lidar2img']
        ida_mat = torch.eye(4).expand(len(imgs), -1, -1).clone()
        ida_mat[:, :3, :3] = post_rots
        ida_mat[:, :2, 2] = post_trans[:, 0:2]

        lidar2img = ida_mat @ torch.from_numpy(np.stack(lidar2img)).to(ida_mat)
        results['lidar2img'] = lidar2img.cpu().numpy()
        results['canvas'] = canvas      # List[(H, W, 3), (H, W, 3), ...]     len = 6
        imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        results['img'] = [imgs[i] for i in range(imgs.shape[0])]
        return sensor2egos, ego2globals, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        """
        Args:
            gt_boxes: (N, 9)
            rotate_angle:
            scale_ratio:
            flip_dx: bool
            flip_dy: bool

        Returns:
            gt_boxes: (N, 9)
            rot_mat: (3, 3）
        """
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:     # 沿着y轴翻转
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:     # 沿着x轴翻转
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)    # 变换矩阵(3, 3)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)     # 变换后的3D框中心坐标
            gt_boxes[:, 3:6] *= scale_ratio    # 变换后的3D框尺寸
            gt_boxes[:, 6] += rotate_angle     # 旋转后的3D框的方位角
            # 翻转也会进一步改变方位角
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['gt_bboxes_3d'], results['gt_labels_3d']      # (N_gt, 9),  (N_gt, )
        gt_boxes = torch.cat([gt_boxes.gravity_center, gt_boxes.tensor[:, 3:]], dim=-1)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        # gt_boxes: (N, 9)  BEV增广变换后的3D框
        # bda_rot: (3, 3)   BEV增广矩阵, 包括旋转、缩放和翻转.
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        sensor2egos, ego2globals, intrins = results['img_inputs'][:3]
        post_rots, post_trans = results['img_inputs'][3:]
        results['img_inputs'] = (sensor2egos, ego2globals, intrins, post_rots,
                                 post_trans, bda_rot)

        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        return results


@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width, img):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        depth = points[:, 2]    # (N_points, )
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )

        # import matplotlib.pyplot as plt
        #
        # mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        # std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
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

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        imgs = results['img'][0]
        depth_map_list = []

        points_camego_aug = results['points'].tensor[:, :3]
        rots, trans, intrins = results['img_inputs'][:3]
        post_rots, post_trans, bda = results['img_inputs'][3:]
        points_camego = points_camego_aug - bda[:3, 3].view(1, 3)
        points_camego = points_camego.matmul(torch.inverse(bda[:3, :3]).T)

        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            lidar2cam = results['cams'][cam_name]['lidar2cam']
            lidar2cam = torch.tensor(lidar2cam).to(points_camego)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2img = cam2img.matmul(lidar2cam.T)

            # (p * l2i.T) * (ida.T)
            # p * (ida * l2i).T
            points_img = points_camego.matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]

            depth_map = self.points2depthmap(points_img, imgs.shape[0],
                                             imgs.shape[1], results['img'][cid])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class LoadMultiTaskAnnotations3D(LoadAnnotations3D):
    def __init__(self,
                 with_tracking_3d=True,
                 with_seg_map_3d=True,
                 with_occ_3d=True,
                 with_agent_3d=True,
                 with_ego_3d=True,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox_3d,
            with_label_3d,
            with_attr_label,
            with_mask_3d,
            with_seg_3d,
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            with_bbox_depth,
            poly2mask,
            seg_3d_dtype,
            file_client_args=file_client_args)

        self.with_tracking_3d = with_tracking_3d
        self.with_seg_map_3d = with_seg_map_3d
        self.with_occ_3d = with_occ_3d
        self.with_agent_3d = with_agent_3d
        self.with_ego_3d = with_ego_3d

    def _load_tracking_3d(self, results):
        results['instance_inds'] = results['ann_info']['instance_inds']
        return results

    def _load_agent_3d(self, results):
        results['gt_agent_fut_trajs'] = results['ann_info']['gt_agent_fut_trajs']
        results['gt_agent_fut_masks'] = results['ann_info']['gt_agent_fut_masks']
        return results

    def _load_ego_3d(self, results):
        results['gt_ego_fut_trajs'] = results['ann_info']['gt_ego_fut_trajs']
        results['gt_ego_fut_masks'] = results['ann_info']['gt_ego_fut_masks']
        # results['gt_ego_fut_cmd'] = results['ann_info']['gt_ego_fut_cmd']
        results['fut_boxes'] = results['ann_info']['fut_boxes']
        return results

    def _load_occ_3d(self, results):
        gt_occ = np.load(results['ann_info']['occ_path'])
        results['gt_occ'] = gt_occ['semantics']
        if 'mask_camera' in gt_occ:
            results['mask_camera'] = gt_occ['mask_camera']
        else:
            results['mask_camera'] = np.zeros_like(results['gt_occ']).astype(bool)
        if 'flow' in gt_occ:
            # results['gt_occ_flow'] = gt_occ['flow'].permute(1,0,2)   #(W,H,C)
            # results['gt_occ'] = results['gt_occ'].permute(1,0,2)  #(W,H,C)
            
            # numpy: (C, H, W) -> (W, H, C)
            # print('###flow shape:', gt_occ['flow'].shape)
            # print('###gt_occ shape:', results['gt_occ'].shape)
            # results['gt_occ_flow'] = gt_occ['flow'].transpose(1,0,2,3)
            # results['gt_occ'] = results['gt_occ'].transpose(1,0,2)
            results['gt_occ_flow'] = gt_occ['flow']

        return results

    def _load_seg_map_3d(self, results):
        results['gt_bev_masks'] = results['ann_info']['gt_bev_masks']
        return results

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_tracking_3d:
            results = self._load_tracking_3d(results)
            if results is None:
                return None

        if self.with_ego_3d:
            results = self._load_ego_3d(results)
            if results is None:
                return None

        if self.with_agent_3d:
            results = self._load_agent_3d(results)
            if results is None:
                return None

        if self.with_seg_map_3d:
            results = self._load_seg_map_3d(results)
            if results is None:
                return None

        if self.with_occ_3d:
            results = self._load_occ_3d(results)
            if results is None:
                return None

        return results
