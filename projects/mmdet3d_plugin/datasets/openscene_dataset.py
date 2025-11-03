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
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from shapely import affinity
from torch.utils.data import Dataset
import pyquaternion
from shapely.geometry import LineString
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose

from .evaluation.occ.occ_eval import OCCEvaluate
from .utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
)


@DATASETS.register_module()
class OpenSceneDataset(Dataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        occ_root=None,
        classes=None,
        map_classes=None,
        occ_class=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        version="v1.0-trainval",
        use_valid_flag=False,
        filter_empty_gt=True,
        vis_score_threshold=0.25,
        sequences_split_num=1,
        with_seq_flag=False,
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

        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None: 
            self.MAP_CLASSES = map_classes
        if occ_class is not None:
            self.OCC_CLASSES = occ_class

        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.with_velocity = with_velocity

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold
        self.sequences_split_num = sequences_split_num

        if with_seq_flag:
            self._set_sequence_group_flag()
        
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
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
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

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        return data

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

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
    
    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = dict(
            token=info["token"],
            map_location=info["map_location"],
            pts_filename=info["lidar_path"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )

        points = LidarPointCloud.from_buffer(open(osp.join(self.data_root, 'sensor_blobs/trainval', info['lidar_path']), 'rb'), "pcd").points
        map_api = get_maps_api(osp.join(self.data_root, 'maps'), "nuplan-maps-v1.0", info['map_location'])
        ego_translation = info["ego2global_translation"]
        ego_quaternion = pyquaternion.Quaternion(info["ego2global_rotation"])
        global_ego_pose = np.array(
            [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
            dtype=np.float64,
        )

        ego_pose = StateSE2(*global_ego_pose)

        bev_semantic_classes = {
            1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
            2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
            3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
            4: (
                "box",
                [
                    TrackedObjectType.CZONE_SIGN,
                    TrackedObjectType.BARRIER,
                    TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT,
                ],
            ),  # static_objects
            5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
            6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
        }

        bev_semantic_map = np.zeros([128, 256], dtype=np.int64)
        for label, (entity_type, layers) in bev_semantic_classes.items():
            if entity_type == "polygon":
                map_object_dict = map_api.get_proximal_map_objects(
                    point=ego_pose.point, radius=[30, 60], layers=layers
                )
                map_polygon_mask = np.zeros([128, 256][::-1], dtype=np.uint8)
                for layer in layers:
                    for map_object in map_object_dict[layer]:
                        polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                        exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                        import pdb;pdb.set_trace()


        import pdb;pdb.set_trace()

        return input_dict

    @staticmethod
    def _geometry_local_coords(geometry, origin: StateSE2):
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry


    def get_ann_info(self, index):
        info = self.data_infos[index]

        
        return
