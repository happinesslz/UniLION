import random
import math
import os
from os import path as osp
import cv2
import tempfile
import copy
import prettytable

import numpy as np
import torch
from mmdet3d.core import get_box_type, LiDARInstance3DBoxes
from torch.utils.data import Dataset
import pyquaternion
from shapely.geometry import LineString
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose

from .evaluation.map.seg_eval import SegEvaluate
from .evaluation.occ.occ_eval import OCCEvaluate
from .map_utils.nuscmap_extractor import NuscMapExtractor
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)


@DATASETS.register_module()
class NuScenes3DDataset(Dataset):
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
    )
    SEG_MAP_CLASSES = (
        'drivable_area',
        'ped_crossing',
        'walkway',
        'stop_line',
        'carpark_area',
        'divider'
    )
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        occ_pipeline=None,
        det_pipeline=None,
        data_root=None,
        occ_root=None,
        classes=None,
        map_classes=None,
        seg_map_config=None,
        occ_class=None,
        bda_aug_conf=None,
        load_interval=1,
        with_velocity=True,
        skip_prob=0.,
        sequence_flip_prob=0.5,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        filter_empty_gt=True,
        vis_score_threshold=0.25,
        sequences_split_num=1,
        map_size=(60, 30),
        keep_consistent_seq_aug=False,
        work_dir=None,
        eval_config=None,
        box_type_3d='LiDAR',
    ):
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.occ_root = os.path.join(data_root, occ_root)
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob

        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None:
            self.MAP_CLASSES = map_classes
        if occ_class is not None:
            self.OCC_CLASSES = occ_class
        if seg_map_config is not None and 'seg_map_classes' in seg_map_config:
            self.SEG_MAP_CLASSES = seg_map_config['seg_map_classes']

        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)
        ori_data_infos = copy.deepcopy(self.data_infos)
        self.ori_num = len(self.data_infos)

        self.seg_map_config = seg_map_config

        self.bda_aug_conf = bda_aug_conf

        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug

        self._set_sequence_group_flag()
        ori_flag = copy.deepcopy(self.flag)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.det_pipeline = det_pipeline
        if det_pipeline is not None:
            self.det_pipeline = Compose(det_pipeline)
            # base + detection
            self.data_infos = self.data_infos + ori_data_infos
            self.det_num = [len(self.data_infos) - len(ori_data_infos), len(self.data_infos)]
            self.flag = np.concatenate([self.flag, ori_flag + max(self.flag) + 1])
        self.occ_pipeline = occ_pipeline
        if occ_pipeline is not None:
            self.occ_pipeline = Compose(occ_pipeline)
            # base + occ
            self.data_infos = self.data_infos + ori_data_infos
            self.occ_num = [len(self.data_infos) - len(ori_data_infos), len(self.data_infos)]
            self.flag = np.concatenate([self.flag, ori_flag + max(self.flag) + 1])

        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.det3d_eval_configs.class_names = list(self.det3d_eval_configs.class_range.keys())
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)
        self.track3d_eval_configs.class_names = list(self.track3d_eval_configs.class_range.keys())
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.gt_cls = ['car', 'truck', 'construction_vehicle',
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
        )
        self.rate = 0.5
        self.rate_vy = 0.2
        self.rate_rotation = -1
        self.thred_vy_by_vx = 1.0

        self.vis_score_threshold = vis_score_threshold

        self.nusc_map_extractor = NuscMapExtractor(self.data_root, map_size)

        self.work_dir = work_dir
        self.eval_config = eval_config

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.sequences_split_num == -1:
            self.flag = np.zeros(len(self.data_infos), dtype=np.int64)
            return

        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]["sweeps"]) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(
                    range(len(self.data_infos)), dtype=np.int64
                )
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag]
                                    / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1
        
                assert len(new_flags) == len(self.flag)
                # assert (
                #     len(np.bincount(new_flags))
                #     == len(np.bincount(self.flag)) * self.sequences_split_num
                # )
                self.flag = np.array(new_flags, dtype=np.int64)

    def pre_pipeline(self, results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def __getitem__(self, data):
        if isinstance(data, dict):
            idx = data['idx']
            bda_cfg = data['bda_cfg']
        else:
            idx = data
            bda_cfg = self.sample_bda_augmentation(idx)
        if self.test_mode:
            return self.prepare_test_data(idx, bda_cfg)
        while True:
            data = self.prepare_train_data(idx, bda_cfg)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        return data

    def prepare_train_data(self, index, bda_cfg):
        input_dict = self.get_data_info(index, bda_cfg)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        if self.det_pipeline is not None and (self.det_num[0] <= index < self.det_num[1]):
            example = self.det_pipeline(input_dict)
            example['img_metas'].data['task'] = ['detection']
        if self.occ_pipeline is not None and (self.occ_num[0] <= index < self.occ_num[1]):
            example = self.occ_pipeline(input_dict)
            example['img_metas'].data['task'] = ['occ']
        if index < self.ori_num:
            example = self.pipeline(input_dict)
            example['img_metas'].data['task'] = []
            if self.det_pipeline is None:
                example['img_metas'].data['task'].append('detection')
            if self.occ_pipeline is None:
                example['img_metas'].data['task'].append('occ')
            example['img_metas'].data['task'].append('map')

        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def prepare_test_data(self, index, bda_cfg):
        input_dict = self.get_data_info(index, bda_cfg)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def _rand_another(self, idx):
        if self.sequences_split_num == -1:
            pool = np.where(self.flag == self.flag[idx])[0]
        else:
            pool = np.where(self.flag != self.flag[idx])[0]
        return np.random.choice(pool)

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata)
        return data_infos

    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms

    def sample_bda_augmentation(self, index):
        """Generate bda augmentation values based on bda_config."""
        if not self.test_mode:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
            vel_aug_dict = dict()
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
            vel_aug_dict = dict()

        bda_cfg = dict()
        bda_cfg['rotate_bda'] = rotate_bda
        bda_cfg['scale_bda'] = scale_bda
        bda_cfg['flip_dx'] = flip_dx
        bda_cfg['flip_dy'] = flip_dy
        bda_cfg['tran_bda'] = tran_bda
        bda_cfg['vel_aug_dict'] = vel_aug_dict

        return bda_cfg

    def map_aug(self, input_dict):
        rotate_bda = input_dict['bda_cfg']['rotate_bda']
        scale_bda = input_dict['bda_cfg']['scale_bda']
        flip_dx = input_dict['bda_cfg']['flip_dx']
        flip_dy = input_dict['bda_cfg']['flip_dy']
        tran_bda = input_dict['bda_cfg']['tran_bda']
     
        rotate_angle = torch.tensor(rotate_bda / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_bda, 0, 0], [0, scale_bda, 0],
                                  [0, 0, scale_bda]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])

        rot_mat = flip_mat @ (scale_mat @ rot_mat)

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = rot_mat

        tran_bda = torch.from_numpy(tran_bda)
        bda_mat[:3, 3] = tran_bda
        bda_mat[:3,:] = flip_mat @ bda_mat[:3,:]
        return bda_mat.numpy()

    def generate_map_annos(self, info, input_dict):
        fce2g_r_mat = pyquaternion.Quaternion(
            info['cams']['CAM_FRONT']["ego2global_rotation"]
        ).rotation_matrix
        fce2g_t = np.array(info['cams']['CAM_FRONT']["ego2global_translation"])

        map_bda_mat = self.map_aug(input_dict)

        # generate map annos
        camego2global = np.eye(4)
        camego2global[:3, :3] = fce2g_r_mat
        camego2global[:3, 3] = fce2g_t

        camego2global = camego2global @ np.linalg.inv(map_bda_mat)

        ce2g_t = camego2global[:3, 3]
        ce2g_r = camego2global[:3, :3]
        map_geoms = self.nusc_map_extractor.get_map_geom(info['map_location'], ce2g_t, ce2g_r)
        map_annos = geom2anno(self.MAP_CLASSES, map_geoms)

        gt_bev_masks = self.nusc_map_extractor.get_map_mask(self.seg_map_config, info['map_location'],
                                                            input_dict['bda_cfg']['scale_bda'], ce2g_t, ce2g_r)
        return map_annos, gt_bev_masks

    def get_data_info(self, index, bda_cfg=None):
        info = self.data_infos[index]
        input_dict = dict(
            token=info["token"],
            map_location=info["map_location"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            scene_token=info["scene_token"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            ego_status=info['ego_status'].astype(np.float32)
        )

        fce2g_r_mat = pyquaternion.Quaternion(
            info['cams']['CAM_FRONT']["ego2global_rotation"]
        ).rotation_matrix
        fce2g_t = np.array(info['cams']['CAM_FRONT']["ego2global_translation"])
        camego2global = np.eye(4)
        camego2global[:3, :3] = fce2g_r_mat
        camego2global[:3, 3] = fce2g_t

        input_dict["lidar2global"] = camego2global

        input_dict['bda_cfg'] = self.sample_bda_augmentation(index) if bda_cfg is None else bda_cfg

        map_annos, gt_bev_masks = self.generate_map_annos(info, input_dict)
        input_dict['map_infos'] = map_annos
        map_geoms = self.anno2geom(map_annos)
        input_dict["map_geoms"] = map_geoms

        input_dict['gt_ego_fut_cmd'] = info['gt_ego_fut_cmd']

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsic = []
            input_dict['cams'] = dict()

            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])

                # obtain lidar from front camera ego to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["cam2camego_rotation"])
                lidar2cam_t = (
                        cam_info["cam2camego_translation"] @ lidar2cam_r.T
                )

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = copy.deepcopy(cam_info['cam_intrinsic'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)

                cam_info['lidar2cam'] = lidar2cam_rt
                cam_info['lidar2img'] = lidar2img_rt
                input_dict['cams'][cam_type] = cam_info

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsic,
                )
            )

        annos = self.get_ann_info(index)
        annos['gt_bev_masks'] = gt_bev_masks
        input_dict['ann_info'] = annos

        if not self.test_mode:
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
            # empty = [len(geom_list) for _, geom_list in map_geoms.items()]
            # if self.filter_empty_gt and sum(empty) == 0:
            #     return None

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

        anns_results['occ_path'] = os.path.join(self.occ_root, info['scene_name'], info['token'], 'labels.npz')

        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int)[mask]
            anns_results["instance_inds"] = instance_inds

        if 'gt_agent_fut_trajs' in info:
            anns_results['gt_agent_fut_trajs'] = info['gt_agent_fut_trajs'][mask]
            anns_results['gt_agent_fut_masks'] = info['gt_agent_fut_masks'][mask]


        if 'gt_ego_fut_trajs' in info:
            anns_results['gt_ego_fut_trajs'] = info['gt_ego_fut_trajs']
            anns_results['gt_ego_fut_masks'] = info['gt_ego_fut_masks']

            # get future box for planning eval
            fut_ts = int(info['gt_ego_fut_masks'].sum())
            fut_boxes = []
            cur_scene_token = info["scene_token"]
            cur_T_global = get_T_global(info)
            for i in range(1, fut_ts + 1):
                fut_info = self.data_infos[index + i]
                fut_scene_token = fut_info["scene_token"]
                if cur_scene_token != fut_scene_token:
                    break
                if self.use_valid_flag:
                    mask = fut_info["valid_flag"]
                else:
                    mask = fut_info["num_lidar_pts"] > 0

                fut_gt_bboxes_3d = fut_info["gt_boxes"][mask]

                fut_T_global = get_T_global(fut_info)
                T_fut2cur = np.linalg.inv(cur_T_global) @ fut_T_global

                center = fut_gt_bboxes_3d[:, :3] @ T_fut2cur[:3, :3].T + T_fut2cur[:3, 3]
                yaw = np.stack([np.cos(fut_gt_bboxes_3d[:, 6]), np.sin(fut_gt_bboxes_3d[:, 6])], axis=-1)
                yaw = yaw @ T_fut2cur[:2, :2].T
                yaw = np.arctan2(yaw[..., 1], yaw[..., 0])

                fut_gt_bboxes_3d[:, :3] = center
                fut_gt_bboxes_3d[:, 6] = yaw
                fut_boxes.append(fut_gt_bboxes_3d)

            anns_results['fut_boxes'] = fut_boxes

        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det, threshold=self.tracking_threshold if tracking else None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="pts_bbox", tracking=False
    ):
        from nuscenes import NuScenes

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False
        )
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        if not tracking:
            from nuscenes.eval.detection.evaluate import NuScenesEval

            nusc_eval = NuScenesEval(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                    ] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        else:
            from nuscenes.eval.tracking.evaluate import TrackingEval

            nusc_eval = TrackingEval(
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            metrics = nusc_eval.main()

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                if name not in ["pts_bbox", "img_bbox"]:
                    continue
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = jsonfile_prefix
                result_files.update(
                    {
                        name: self._format_bbox(
                            results_, tmp_file_, tracking=tracking
                        )
                    }
                )
        return result_files, tmp_dir

    def format_map_results(self, results, prefix=None):
        submissions = {'results': {},}

        for j, pred in enumerate(results):
            '''
            For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
            'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                contain all vectors predicted in this sample.
            'scores: List of score(float), 
                contain scores of all instances in this sample.
            'labels': List of label(int), 
                contain labels of all instances in this sample.
            '''
            if pred is None: # empty prediction
                continue
            pred = pred['pts_map']

            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = self.data_infos[j]['token']
            for i in range(len(pred['scores'])):
                score = pred['scores'][i].numpy()
                label = pred['labels'][i].numpy()
                vector = pred['vectors'][i].numpy()

                # A line should have >=2 points
                if len(vector) < 2:
                    continue

                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)

            submissions['results'][token] = single_case

        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def format_occ_results(self, results, prefix=None):
        submissions = {'results': {}, }

        for j, pred in enumerate(results):
            if pred is None:  # empty prediction
                continue
            pred = pred['pts_occ']

            token = self.data_infos[j]['token']
            submissions['results'][token] = pred['occ']

        out_path = osp.join(prefix, 'submission_occ.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def format_planning_results(self, results, prefix=None):
        submissions = {'results': {}, }

        for j, pred in enumerate(results):
            if pred is None:  # empty prediction
                continue
            pred = pred['pts_plan_reg']

            token = self.data_infos[j]['token']
            submissions['results'][token] = pred['plan_reg']

        out_path = osp.join(prefix, 'submission_palnning.json')
        print(f'saving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def format_motion_results(self, results, jsonfile_prefix=None, tracking=False, thresh=None):
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det['pts_bbox'], threshold=None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
                filter_with_cls_range=False,
            )
            for i, box in enumerate(boxes):
                if thresh is not None and box.score < thresh:
                    continue
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenes3DDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )
                nusc_anno.update(
                    dict(
                        trajs=det['pts_bbox']['trajs_3d'][i].numpy(),
                    )
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        return nusc_submissions

    def _evaluate_single_motion(self,
                         results,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from .evaluation.motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion

        output_dir = result_path
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEvalMotion(
            nusc,
            config=copy.deepcopy(self.det3d_eval_configs),
            result_path=results,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            seconds=6)
        metrics = nusc_eval.main(render_curves=False)

        MOTION_METRICS = ['EPA', 'min_ade_err', 'min_fde_err', 'miss_rate_err']
        class_names = ['car', 'pedestrian']

        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row_data = [class_name]
            for m in MOTION_METRICS:
                row_data.append('%.4f' % metrics[f'{class_name}_{m}'])
            table.add_row(row_data)
        print_log('\n'+str(table), logger=logger)
        return metrics

    def evaluate(
        self,
        results,
        eval_mode,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        res_path = "results.pkl" if "trainval" in self.version else "results_mini.pkl"
        res_path = osp.join(self.work_dir, res_path)
        print('All Results write to', res_path)
        mmcv.dump(results, res_path)

        results_dict = dict()
        if eval_mode['with_det']:
            self.tracking = eval_mode["with_tracking"]
            self.tracking_threshold = eval_mode["tracking_threshold"]
            for metric in ["detection", "tracking"]:
                tracking = metric == "tracking"
                if tracking and not self.tracking:
                    continue
                result_files, tmp_dir = self.format_results(
                    results, jsonfile_prefix=self.work_dir, tracking=tracking
                )

                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(
                            result_files[name], tracking=tracking
                        )
                    results_dict.update(ret_dict)
                elif isinstance(result_files, str):
                    ret_dict = self._evaluate_single(
                        result_files, tracking=tracking
                    )
                    results_dict.update(ret_dict)

                if tmp_dir is not None:
                    tmp_dir.cleanup()

        if eval_mode['with_occ']:
            self.occ_evaluator = OCCEvaluate(self.data_infos, self.occ_root, self.OCC_CLASSES)
            occ_results_dict = self.occ_evaluator.evaluate(res_path, logger=logger)
            results_dict.update(occ_results_dict)

        if eval_mode['with_map']:
            from .evaluation.map.vector_eval import VectorEvaluate
            if 'pts_seg_map' in results[0]:
                self.seg_map_evaluator = SegEvaluate(self.eval_config)
                seg_map_results_dict = self.seg_map_evaluator.evaluate(res_path, logger=logger)
                results_dict.update(seg_map_results_dict)

            if 'pts_map' in results[0]:
                self.map_evaluator = VectorEvaluate(self.eval_config)
                result_path = self.format_map_results(results, prefix=self.work_dir)
                map_results_dict = self.map_evaluator.evaluate(result_path, logger=logger)
                results_dict.update(map_results_dict)

        if eval_mode['with_motion']:
            thresh = eval_mode["motion_threshhold"]
            result_files = self.format_motion_results(results, jsonfile_prefix=self.work_dir, thresh=thresh)
            motion_results_dict = self._evaluate_single_motion(result_files, self.work_dir, logger=logger)
            results_dict.update(motion_results_dict)

        if eval_mode['with_planning']:
            from .evaluation.planning.planning_eval import planning_eval
            result_path = self.format_planning_results(results, prefix=self.work_dir)
            planning_results_dict = planning_eval(result_path, self.eval_config, logger=logger)
            results_dict.update(planning_results_dict)

        if show or out_dir:
            self.show(results, save_dir=out_dir, show=show, pipeline=pipeline)

        # print main metrics for recording
        metric_str = '\n'
        if "pts_bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("pts_bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("pts_bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("pts_bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("pts_bbox_NuScenes/mAOE"):.4f}\n'
            metric_str += f'mAVE: {results_dict.get("pts_bbox_NuScenes/mAVE"):.4f}\n'
            metric_str += f'mAAE: {results_dict.get("pts_bbox_NuScenes/mAAE"):.4f}\n'
            metric_str += f'NDS: {results_dict.get("pts_bbox_NuScenes/NDS"):.4f}\n\n'

        if "pts_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["pts_bbox_NuScenes/amota"]:.4f}\n'
            metric_str += f'AMOTP: {results_dict["pts_bbox_NuScenes/amotp"]:.4f}\n'
            metric_str += f'RECALL: {results_dict["pts_bbox_NuScenes/recall"]:.4f}\n'
            metric_str += f'MOTAR: {results_dict["pts_bbox_NuScenes/motar"]:.4f}\n'
            metric_str += f'MOTA: {results_dict["pts_bbox_NuScenes/mota"]:.4f}\n'
            metric_str += f'MOTP: {results_dict["pts_bbox_NuScenes/motp"]:.4f}\n'
            metric_str += f'IDS: {results_dict["pts_bbox_NuScenes/ids"]}\n\n'

        if "RayIoU" in results_dict:
            metric_str += f'RayIoU= {results_dict["RayIoU"]:.4f}\n'
            metric_str += f'RayIoU@1= {results_dict["RayIoU@1"]:.4f}\n'
            metric_str += f'RayIoU@2= {results_dict["RayIoU@2"]:.4f}\n'
            metric_str += f'RayIoU@4= {results_dict["RayIoU@4"]:.4f}\n\n'

        if "mAP_normal" in results_dict:
            metric_str += f'ped_crossing= {results_dict["ped_crossing"]:.4f}\n'
            metric_str += f'divider= {results_dict["divider"]:.4f}\n'
            metric_str += f'boundary= {results_dict["boundary"]:.4f}\n'
            metric_str += f'mAP_normal= {results_dict["mAP_normal"]:.4f}\n\n'

        if "mIoU@max" in results_dict:
            metric_str += f'mIoU@max= {results_dict["mIoU@max"]:.4f}\n\n'

        if "car_EPA" in results_dict:
            metric_str += f'Car / Ped\n'
            metric_str += f'epa= {results_dict["car_EPA"]:.4f} / {results_dict["pedestrian_EPA"]:.4f}\n'
            metric_str += f'ade= {results_dict["car_min_ade_err"]:.4f} / {results_dict["pedestrian_min_ade_err"]:.4f}\n'
            metric_str += f'fde= {results_dict["car_min_fde_err"]:.4f} / {results_dict["pedestrian_min_fde_err"]:.4f}\n'
            metric_str += f'mr= {results_dict["car_miss_rate_err"]:.4f} / {results_dict["pedestrian_miss_rate_err"]:.4f}\n\n'

        if "L2" in results_dict:
            metric_str += f'obj_box_col: {(results_dict["obj_box_col"]*100):.3f}%\n'
            metric_str += f'L2: {results_dict["L2"]:.4f}\n\n'

        print_log(metric_str, logger=logger)
        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        save_dir = "./" if save_dir is None else save_dir
        save_dir = os.path.join(save_dir, "visual")
        print_log(os.path.abspath(save_dir))
        pipeline = Compose(pipeline)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videoWriter = None

        for i, result in enumerate(results):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            data_info = pipeline(self.get_data_info(i))
            imgs = []

            raw_imgs = data_info["img"]
            lidar2img = data_info["img_metas"].data["lidar2img"]
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > self.vis_score_threshold
            ]
            if "instance_ids" in result and self.tracking:
                color = []
                for id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                color = []
                for id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[id])
            else:
                color = (255, 0, 0)

            # ===== draw boxes_3d to images =====
            for j, img_origin in enumerate(raw_imgs):
                img = img_origin.copy()
                if len(pred_bboxes_3d) != 0:
                    img = draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                imgs.append(img)

            # ===== draw boxes_3d to BEV =====
            bev = draw_lidar_bbox3d_on_bev(
                pred_bboxes_3d,
                bev_size=img.shape[0] * 2,
                color=color,
            )

            # ===== put text and concat =====
            for j, name in enumerate(
                [
                    "front",
                    "front right",
                    "front left",
                    "rear",
                    "rear left",
                    "rear right",
                ]
            ):
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    (440, 80),
                    color=(255, 255, 255),
                    thickness=-1,
                )
                w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                text_x = int(220 - w / 2)
                text_y = int(40 + h / 2)

                imgs[j] = cv2.putText(
                    imgs[j],
                    name,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            image = np.concatenate(
                [
                    np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                    np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                ],
                axis=0,
            )
            image = np.concatenate([image, bev], axis=1)

            # ===== save video =====
            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    os.path.join(save_dir, "video.avi"),
                    fourcc,
                    7,
                    image.shape[:2][::-1],
                )
            cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image)
            videoWriter.write(image)
        videoWriter.release()


def output_to_nusc_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"]  # .numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
    filter_with_cls_range=True,
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        # box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        # box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        if filter_with_cls_range:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['cams']['CAM_FRONT']["ego2global_rotation"]))
        box.translate(np.array(info['cams']['CAM_FRONT']["ego2global_translation"]))
        box_list.append(box)
    return box_list


def get_T_global(info):
    ego2global = np.eye(4)
    ego2global[:3, :3] = pyquaternion.Quaternion(
        info['cams']['CAM_FRONT']["ego2global_rotation"]
    ).rotation_matrix
    ego2global[:3, 3] = np.array(info['cams']['CAM_FRONT']["ego2global_translation"])
    return ego2global

def geom2anno(map_classes, map_geoms):
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls in map_classes:
            label = map_classes.index(cls)
            vectors[label] = []
            for geom in geom_list:
                line = np.array(geom.coords)
                vectors[label].append(line)
    return vectors