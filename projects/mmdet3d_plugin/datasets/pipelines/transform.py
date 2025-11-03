import warnings
from copy import deepcopy

import torch
import numpy as np
import mmcv
import cv2

from mmdet.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg
from mmcv.image.photometric import imnormalize
from mmdet3d.core import LiDARInstance3DBoxes, DepthInstance3DBoxes, CameraInstance3DBoxes
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines import Compose
from pyquaternion import Quaternion
from torchvision.transforms.functional import rotate


@PIPELINES.register_module()
class ToEgo(object):
    def __init__(self, ego_cam='CAM_FRONT', ):
        self.ego_cam = ego_cam

    def __call__(self, results):
        lidar2lidarego = np.eye(4, dtype=np.float32)
        lidar2lidarego[:3, :3] = Quaternion(
            results['lidar2ego_rotation']).rotation_matrix
        lidar2lidarego[:3, 3] = results['lidar2ego_translation']

        lidarego2global = np.eye(4, dtype=np.float32)
        lidarego2global[:3, :3] = Quaternion(
            results['ego2global_rotation']).rotation_matrix
        lidarego2global[:3, 3] = results['ego2global_translation']

        camego2global = np.eye(4, dtype=np.float32)
        camego2global[:3, :3] = Quaternion(
            results['cams'][self.ego_cam]
            ['ego2global_rotation']).rotation_matrix
        camego2global[:3, 3] = results['cams'][self.ego_cam][
            'ego2global_translation']
        lidar2camego = np.linalg.inv(camego2global) @ lidarego2global @ lidar2lidarego

        points = results['points'].tensor.numpy()
        points_ego = lidar2camego[:3, :3].reshape(1, 3, 3) @ \
                     points[:, :3].reshape(-1, 3, 1) + \
                     lidar2camego[:3, 3].reshape(1, 3, 1)
        points[:, :3] = points_ego.squeeze(-1)
        points = results['points'].new_point(points)
        results['points'] = points
        results['lidar2global'] = camego2global
        return results


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class BEVAug(object):
    def __init__(self, bda_aug_conf, classes, re_aug=False, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.re_aug = re_aug
        self.classes = classes

    def sample_bda_augmentation(self, results):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train and self.re_aug:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = results['bda_cfg']['rotate_bda']
            scale_bda = results['bda_cfg']['scale_bda']
            flip_dx = results['bda_cfg']['flip_dx']
            flip_dy = results['bda_cfg']['flip_dy']
            tran_bda = results['bda_cfg']['tran_bda']

        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy, tran_bda):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                    rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                         6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                    rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, :3] = gt_boxes[:, :3] + tran_bda
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes = results['gt_bboxes_3d'].tensor
        gt_boxes[:, 2] = gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = \
            self.sample_bda_augmentation(results)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1

        # step 1: gt bboxes aug
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy, tran_bda)
        # step 2: input point clouds aug
        if 'points' in results:
            points = results['points'].tensor
            points_aug = (bda_rot @ points[:, :3].unsqueeze(-1)).squeeze(-1)
            points[:, :3] = points_aug + tran_bda
            points = results['points'].new_point(points)
            results['points'] = points

        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        if 'img_inputs' in results:
            rots, trans, intrins = results['img_inputs'][:3]
            post_rots, post_trans = results['img_inputs'][3:]
            results['img_inputs'] = (rots, trans, intrins, post_rots,
                                     post_trans, bda_mat)

        # step 3: gt map aug
        if 'gt_bev_masks' in results:
            if flip_dx:
                results['gt_bev_masks'] = results['gt_bev_masks'][..., :, ::-1].copy()
            if flip_dy:
                results['gt_bev_masks'] = results['gt_bev_masks'][..., ::-1, :].copy()

        # step 4: gt occ aug
        if 'gt_occ' in results:
            if rotate_bda != 0:
                gt_occ = torch.from_numpy(results['gt_occ']).permute(2, 0, 1)  # [16, 200, 200]
                gt_occ = rotate(gt_occ, rotate_bda, fill=255).permute(1, 2, 0)  # [200, 200, 16]
                results['gt_occ'] = gt_occ.numpy()
                
            # gt_occ_flow
            if 'gt_occ_flow' in results:
                gt_occ_flow = torch.from_numpy(results['gt_occ_flow'])  # [200, 200, 16, 2]
                
                # Step 1: First rotate velocity vector directions
                angle_rad = torch.tensor(rotate_bda * torch.pi / 180.0)
                cos_angle = torch.cos(angle_rad)
                sin_angle = torch.sin(angle_rad)
                
                vx_old = gt_occ_flow[..., 0]  # [200, 200, 16]
                vy_old = gt_occ_flow[..., 1]  # [200, 200, 16]
                
                # Rotate velocity vectors at each position
                vx_rotated = vx_old * cos_angle - vy_old * sin_angle
                vy_rotated = vx_old * sin_angle + vy_old * cos_angle
                
                # Step 2: Then rotate spatial positions
                vx_spatial = rotate(vx_rotated.permute(2, 0, 1), rotate_bda, fill=0).permute(1, 2, 0)
                vy_spatial = rotate(vy_rotated.permute(2, 0, 1), rotate_bda, fill=0).permute(1, 2, 0)
                
                results['gt_occ_flow'] = torch.stack([vx_spatial, vy_spatial], dim=-1).numpy()
                
            if flip_dx:
                results['gt_occ'] = results['gt_occ'][::-1, ...].copy()
                # results['mask_lidar'] = results['mask_lidar'][::-1, ...].copy()
                results['mask_camera'] = results['mask_camera'][::-1, ...].copy()
                
                if 'gt_occ_flow' in results:
                    results['gt_occ_flow'] = results['gt_occ_flow'][::-1, ...].copy()
                    results['gt_occ_flow'][..., 0] = -results['gt_occ_flow'][..., 0]
                    
            if flip_dy:
                results['gt_occ'] = results['gt_occ'][:, ::-1, ...].copy()
                # results['mask_lidar'] = results['mask_lidar'][:, ::-1, ...].copy()
                results['mask_camera'] = results['mask_camera'][:, ::-1, ...].copy()
                
                if 'gt_occ_flow' in results:
                    results['gt_occ_flow'] = results['gt_occ_flow'][:, ::-1, ...].copy()
                    results['gt_occ_flow'][..., 1] = -results['gt_occ_flow'][..., 1]
                
        # step 5: planning aug
        if 'lidar2global' in results:
            results['lidar2global'] = results['lidar2global'] @ np.linalg.inv(bda_mat)
        return results


@PIPELINES.register_module()
class VelocityAug(object):
    def __init__(self, rate=0.5, rate_vy=0.2, rate_rotation=-1, speed_range=None, thred_vy_by_vx=1.0,
                 ego_cam='CAM_FRONT'):
        # must be identical to that in tools/create_data_bevdet.py
        self.cls = ['car', 'truck', 'construction_vehicle',
                    'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle',
                    'pedestrian', 'traffic_cone']
        self.speed_range = dict(
            car=[-10, 30, 6],
            truck=[-10, 30, 6],
            construction_vehicle=[-10, 30, 3],
            bus=[-10, 30, 3],
            trailer=[-10, 30, 3],
            barrier=[-5, 5, 3],
            motorcycle=[-2, 25, 3],
            bicycle=[-2, 15, 2],
            pedestrian=[-1, 10, 2]
        ) if speed_range is None else speed_range
        self.rate = rate
        self.thred_vy_by_vx=thred_vy_by_vx
        self.rate_vy = rate_vy
        self.rate_rotation = rate_rotation
        self.ego_cam = ego_cam

    def interpolating(self, vx, vy, delta_t, box, rot):
        delta_t_max = np.max(delta_t)
        if vy ==0 or vx == 0:
            delta_x = delta_t*vx
            delta_y = np.zeros_like(delta_x)
            rotation_interpolated = np.zeros_like(delta_x)
        else:
            theta = np.arctan2(abs(vy), abs(vx))
            rotation = 2 * theta
            radius = 0.5 * delta_t_max * np.sqrt(vx ** 2 + vy ** 2) / np.sin(theta)
            rotation_interpolated = delta_t / delta_t_max * rotation
            delta_y = radius - radius * np.cos(rotation_interpolated)
            delta_x = radius * np.sin(rotation_interpolated)
            if vy<0:
                delta_y = - delta_y
            if vx<0:
                delta_x = - delta_x
            if np.logical_xor(vx>0, vy>0):
                rotation_interpolated = -rotation_interpolated
        aug = np.zeros((delta_t.shape[0],3,3), dtype=np.float32)
        aug[:, 2, 2] = 1.
        sin = np.sin(-rotation_interpolated)
        cos = np.cos(-rotation_interpolated)
        aug[:,:2,:2] = np.stack([cos,sin,-sin,cos], axis=-1).reshape(delta_t.shape[0], 2, 2)
        aug[:,:2, 2] = np.stack([delta_x, delta_y], axis=-1)

        corner2center = np.eye(3)
        corner2center[0, 2] = -0.5 * box[3]

        instance2ego = np.eye(3)
        yaw = -box[6]
        s = np.sin(yaw)
        c = np.cos(yaw)
        instance2ego[:2,:2] = np.stack([c,s,-s,c]).reshape(2,2)
        instance2ego[:2,2] = box[:2]
        corner2ego = instance2ego @ corner2center
        corner2ego = corner2ego[None, ...]
        if not rot == 0:
            t_rot = np.eye(3)
            s_rot = np.sin(-rot)
            c_rot = np.cos(-rot)
            t_rot[:2,:2] = np.stack([c_rot, s_rot, -s_rot, c_rot]).reshape(2,2)

            instance2ego_ = np.eye(3)
            yaw_ = -box[6] - rot
            s_ = np.sin(yaw_)
            c_ = np.cos(yaw_)
            instance2ego_[:2, :2] = np.stack([c_, s_, -s_, c_]).reshape(2, 2)
            instance2ego_[:2, 2] = box[:2]
            corner2ego_ = instance2ego_ @ corner2center
            corner2ego_ = corner2ego_[None, ...]
            t_rot = instance2ego @ t_rot @ np.linalg.inv(instance2ego)
            aug = corner2ego_ @ aug @ np.linalg.inv(corner2ego_) @ t_rot[None, ...]
        else:
            aug = corner2ego @ aug @ np.linalg.inv(corner2ego)
        return aug

    def __call__(self, results):
        gt_boxes = results['gt_bboxes_3d'].tensor.numpy().copy()
        gt_velocity = gt_boxes[:,7:]
        gt_velocity_norm = np.sum(np.square(gt_velocity), axis=1)
        points = results['points'].tensor.numpy().copy()
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        vel_aug_dict = results['bda_cfg']['vel_aug_dict']

        for bid in range(gt_boxes.shape[0]):
            cls = self.cls[results['gt_labels_3d'][bid]]
            track_id = results['instance_inds'][bid]

            aug_dict = vel_aug_dict[track_id] if track_id in vel_aug_dict else None
            aug_dict = None

            points_all = points[point_indices[:, bid]]
            delta_t = np.unique(points_all[:, 4])
            aug_rate_cls = self.rate if isinstance(self.rate, float) else self.rate[cls]

            mask = np.random.rand() > aug_rate_cls if aug_dict is None else aug_dict['mask']

            if points_all.shape[0] == 0 or \
                    delta_t.shape[0] < 3 or \
                    gt_velocity_norm[bid] > 0.01 or \
                    cls not in self.speed_range or \
                    mask:
                continue

            if aug_dict is not None:
                vx = aug_dict['vx']
                vy = aug_dict['vy']
                rot = aug_dict['rot']
            else:
                # sampling speed vx,vy in instance coordinate
                vx = np.random.rand() * (self.speed_range[cls][1] -
                                        self.speed_range[cls][0]) + \
                    self.speed_range[cls][0]
                if np.random.rand() < self.rate_vy:
                    max_vy = min(self.speed_range[cls][2] * 2, abs(vx) * self.thred_vy_by_vx)
                    vy = (np.random.rand() - 0.5) * max_vy
                else:
                    vy = 0.0
                vx = -vx

                rot = 0.0
                if np.random.rand() < self.rate_rotation:
                    rot = (np.random.rand() - 0.5) * 1.57

                vel_aug_dict[track_id] = dict(vx=vx, vy=vy, rot=rot, mask=mask)

            aug = self.interpolating(vx, vy, delta_t, gt_boxes[bid], rot)

            # update rotation
            gt_boxes[bid, 6] += rot

            # update velocity
            delta_t_max = np.max(delta_t)
            delta_t_max_index = np.argmax(delta_t)
            center = gt_boxes[bid:bid + 1, :2]
            center_aug = center @ aug[delta_t_max_index, :2, :2].T + aug[delta_t_max_index, :2, 2]
            vel = (center - center_aug) / delta_t_max
            gt_boxes[bid, 7:] = vel

            # update points
            for fid in range(delta_t.shape[0]):
                points_curr_frame_idxes = points_all[:,4] == delta_t[fid]

                points_all[points_curr_frame_idxes, :2] = \
                    points_all[points_curr_frame_idxes, :2]  @ aug[fid,:2,:2].T + aug[fid,:2, 2:3].T
            points[point_indices[:, bid]] = points_all


        results['points'] = results['points'].new_point(points)
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'].new_box(gt_boxes)
        return results


@PIPELINES.register_module()
class MultiObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['gt_agent_fut_trajs'] = input_dict['gt_agent_fut_trajs'][mask.numpy().astype(np.bool)]
        input_dict['gt_agent_fut_masks'] = input_dict['gt_agent_fut_masks'][mask.numpy().astype(np.bool)]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class MultiObjectNameFilter(object):
    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        input_dict['gt_agent_fut_trajs'] = input_dict['gt_agent_fut_trajs'][gt_bboxes_mask]
        input_dict['gt_agent_fut_masks'] = input_dict['gt_agent_fut_masks'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class UnifiedObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, sample_method='depth', modify_points=False, mixup_rate=-1):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.sample_method = sample_method
        self.modify_points = modify_points
        self.mixup_rate = mixup_rate
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                with_img=True)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=False)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_points_idx = sampled_dict["points_idx"]
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points_idx = -1 * np.ones(len(points), dtype=np.int)
            # check the points dimension
            # points = points.cat([sampled_points, points])
            points = points.cat([points, sampled_points])
            # points = sampled_points

            points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)

            if self.sample_2d:
                imgs = input_dict['img']
                lidar2img = input_dict['lidar2img']
                sampled_img = sampled_dict['images']
                sampled_img_norm = list()
                for i in range(len(sampled_img)):
                    if len(sampled_img[i]) == 0:
                        sampled_img_norm.append(sampled_img[i])
                    else:
                        sampled_img_norm.append(self.normalize_img(sampled_img[i]))
                sampled_img = sampled_img_norm

                sampled_num = len(sampled_gt_bboxes_3d)
                imgs, points_keep = self.unified_sample(imgs, lidar2img,
                                                        points.tensor.numpy(),
                                                        points_idx, gt_bboxes_3d.corners.numpy(),
                                                        sampled_img, sampled_num)

                input_dict['img'] = imgs

                if self.modify_points:
                    points = points[points_keep]

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict

    def normalize_img(self, img):
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        to_rgb = True
        img = imnormalize(np.array(img), mean, std, to_rgb)
        return img

    def unified_sample(self, imgs, lidar2img, points, points_idx, bboxes_3d, sampled_img, sampled_num):
        # for boxes
        bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)
        is_raw = np.ones(len(bboxes_3d))
        is_raw[-sampled_num:] = 0
        is_raw = is_raw.astype(bool)
        raw_num = len(is_raw) - sampled_num
        # for point cloud
        points_3d = points[:, :4].copy()
        points_3d[:, -1] = 1
        points_keep = np.ones(len(points_3d)).astype(np.bool)
        new_imgs = imgs

        assert len(imgs) == len(lidar2img) and len(sampled_img) == sampled_num
        for _idx, (_img, _lidar2img) in enumerate(zip(imgs, lidar2img)):
            coord_img = bboxes_3d @ _lidar2img.T
            coord_img[..., :2] /= coord_img[..., 2, None]
            depth = coord_img[..., 2]
            img_mask = (depth > 0).all(axis=-1)
            img_count = img_mask.nonzero()[0]
            if img_mask.sum() == 0:
                continue
            depth = depth.mean(1)[img_mask]
            coord_img = coord_img[..., :2][img_mask]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:, 0::2] = np.clip(bbox[:, 0::2], a_min=0, a_max=_img.shape[1] - 1)
            bbox[:, 1::2] = np.clip(bbox[:, 1::2], a_min=0, a_max=_img.shape[0] - 1)
            img_mask = ((bbox[:, 2:] - bbox[:, :2]) > 1).all(axis=-1)
            if img_mask.sum() == 0:
                continue
            depth = depth[img_mask]
            if 'depth' in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)
            img_count = img_count[img_mask][paste_order]
            bbox = bbox[img_mask][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int)
            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1]:_box[3], _box[0]:_box[2]])

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = raw_img.pop(0)
                    else:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = \
                            _img[_box[1]:_box[3], _box[0]:_box[2]] * (1 - self.mixup_rate) + raw_img.pop(
                                0) * self.mixup_rate
                    fg_mask[_box[1]:_box[3], _box[0]:_box[2]] = 1
                else:
                    img_crop = sampled_img[_count - raw_num]
                    if len(img_crop) == 0: continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2, 3]] - _box[[0, 1]]))
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = img_crop
                    else:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = \
                            _img[_box[1]:_box[3], _box[0]:_box[2]] * (1 - self.mixup_rate) + img_crop * self.mixup_rate

                paste_mask[_box[1]:_box[3], _box[0]:_box[2]] = _count

            new_imgs[_idx] = _img

            # calculate modify mask
            if self.modify_points:
                points_img = points_3d @ _lidar2img.T
                points_img[:, :2] /= points_img[:, 2, None]
                depth = points_img[:, 2]
                img_mask = depth > 0
                if img_mask.sum() == 0:
                    continue
                img_mask = (points_img[:, 0] > 0) & (points_img[:, 0] < _img.shape[1]) & \
                           (points_img[:, 1] > 0) & (points_img[:, 1] < _img.shape[0]) & img_mask
                points_img = points_img[img_mask].astype(int)
                new_mask = paste_mask[points_img[:, 1], points_img[:, 0]] == (points_idx[img_mask] + raw_num)
                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_num)
                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                raw_mask = raw_fg[points_img[:, 1], points_img[:, 0]] | raw_bg[points_img[:, 1], points_img[:, 0]]
                keep_mask = new_mask | raw_mask
                points_keep[img_mask] = points_keep[img_mask] & keep_mask

        return new_imgs, points_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class TestTimeAug3D(object):
    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else [float(pts_scale_ratio)]

        assert mmcv.is_list_of(self.img_scale, tuple)
        assert mmcv.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            # results.copy will cause bug
                            # since it is shallow copy
                            _results = deepcopy(results)

                            bda_cfg = {
                                'rotate_bda': 0,
                                'scale_bda': pts_scale_ratio,
                                'flip_dx': pcd_vertical_flip,
                                'flip_dy': pcd_horizontal_flip,
                                'tran_bda': np.array([[0., 0., 0.]])
                            }
                            _results['bda_cfg'] = bda_cfg
                            data = self.transforms(_results)
                            aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
