import json
import os
import pickle
import math
import copy
import argparse
import yaml
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

import mmcv

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.geometry_utils import transform_matrix
from create_gt_database import create_groundtruth_database
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global

from projects.mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i - 1] < utimes[i] - utime):
        i -= 1
    return i


def geom2anno(map_geoms):
    MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
    )
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls in MAP_CLASSES:
            label = MAP_CLASSES.index(cls)
            vectors[label] = []
            for geom in geom_list:
                line = np.array(geom.coords)
                vectors[label].append(line)
    return vectors


def _fill_trainval_infos(root_path, train_scenes, val_scenes):
    train_nusc_infos = []
    val_nusc_infos = []

    for train_scene in mmcv.track_iter_progress(train_scenes):
        train_scene_meta = osp.join('meta_datas', 'trainval', train_scene + '.pkl')
        train_scene_meta = mmcv.load(osp.join(root_path, train_scene_meta), file_format="pkl")
        train_nusc_infos.extend(train_scene_meta)

    for val_scene in mmcv.track_iter_progress(val_scenes):
        val_scene_meta = osp.join('meta_datas', 'trainval', val_scene + '.pkl')
        val_scene_meta = mmcv.load(osp.join(root_path, val_scene_meta), file_format="pkl")
        val_nusc_infos.extend(val_scene_meta)

    return train_nusc_infos, val_nusc_infos


def create_openscene_infos(root_path,
                           out_path,
                           split_path,
                           info_prefix,
                           version=1.1,
                           roi_size=(30, 60)):
    splits = json.load(open(split_path, 'r', encoding='utf-8'))
    train_scenes = splits['train_logs']
    val_scenes = splits['val_logs']
    # test_scenes = splits['test_logs']
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(root_path, train_scenes, val_scenes)

    metadata = dict(version=version)

    print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = osp.join(out_path, '{}_infos_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    data['infos'] = val_nusc_infos
    info_val_path = osp.join(out_path, '{}_infos_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)


def openscene_data_prep(root_path,
                        split_path,
                        info_prefix,
                        dataset_name,
                        out_dir,
                        version='1.1'):
    create_openscene_infos(
        root_path, out_dir, split_path, info_prefix, version=version)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--split-path',
    type=str,
    default='',
    help='specify the split path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.1',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'openscene':
        train_version = f'{args.version}-trainval'
        openscene_data_prep(
            root_path=args.root_path,
            split_path=args.split_path,
            info_prefix=args.extra_tag,
            dataset_name='OpenSceneDataset',
            out_dir=args.out_dir,
            version=train_version)
        # test_version = f'{args.version}-test'
        # openscene_data_prep(
        #     root_path=args.root_path,
        #     split_path=args.split_path,
        #     info_prefix=args.extra_tag,
        #     dataset_name='OpenSceneDataset',
        #     out_dir=args.out_dir,
        #     version=test_version)
