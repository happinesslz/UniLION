import pickle
import copy
import prettytable
from mmcv.utils import print_log
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory as det_configs
from projects.mmdet3d_plugin.datasets.evaluation.motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion


from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Polygon

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import build_dataset, build_dataloader

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners



def check_collision(ego_box, boxes):
    '''
        ego_box: tensor with shape [7], [x, y, z, w, l, h, yaw]
        boxes: tensor with shape [N, 7]
    '''
    if  boxes.shape[0] == 0:
        return False

    # follow uniad, add a 0.5m offset
    ego_box[0] += (0.985793 + 0.5) * torch.cos(ego_box[6])
    ego_box[1] += (0.985793 + 0.5) * torch.sin(ego_box[6])
    # ego_box[0] += 0.985793# * torch.cos(ego_box[6])
    # ego_box[0] += 0.5 * torch.cos(ego_box[6])
    # ego_box[1] += 0.5 * torch.sin(ego_box[6])
    # ego_box[0] += 0.5
    # ego_box[1] += 0.5
    ego_corners_box = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4, 0], :2]
    corners_box = box3d_to_corners(boxes)[:, [0, 3, 7, 4, 0], :2]
    ego_poly = Polygon([(point[0], point[1]) for point in ego_corners_box[:-1]])

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 1, figsize=(20, 20))
    # axes.set_xlim(- 40, 40)
    # axes.set_ylim(- 40, 40)
    # axes.axis('off')
    #
    # x = ego_corners_box[:, 0]
    # y = ego_corners_box[:, 1]
    # axes.plot(x, y, color=[1, 0, 0], linewidth=3, linestyle='-')
    #
    # for corner in corners_box:
    #     x = corner[:, 0]
    #     y = corner[:, 1]
    #     axes.plot(x, y, color=[0, 0, 0], linewidth=3, linestyle='-')
    #
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig('/data/jhhou/disk/temp/temp.jpg')
    # import pdb;
    # pdb.set_trace()

    for i in range(len(corners_box)):
        box_poly = Polygon([(point[0], point[1]) for point in corners_box[:, :-1][i]])
        collision = ego_poly.intersects(box_poly)
        if collision:
            return True

    return False

def get_yaw(traj):
    start = traj[0]
    end = traj[-1]
    dist = torch.linalg.norm(end - start, dim=-1)
    if dist < 1.0:
        # return traj.new_ones(traj.shape[0]) * np.pi / 2
        return traj.new_zeros(traj.shape[0])

    zeros = traj.new_zeros((1, 2))
    traj_cat = torch.cat([zeros, traj], dim=0)
    yaw = traj.new_zeros(traj.shape[0]+1)
    yaw[..., 1:-1] = torch.atan2(
        traj_cat[..., 2:, 1] - traj_cat[..., :-2, 1],
        traj_cat[..., 2:, 0] - traj_cat[..., :-2, 0],
    )
    yaw[..., -1] = torch.atan2(
        traj_cat[..., -1, 1] - traj_cat[..., -2, 1],
        traj_cat[..., -1, 0] - traj_cat[..., -2, 0],
    )
    return yaw[:-1]

class PlanningMetric():
    def __init__(
        self,
        n_future=6,
        compute_on_step: bool = False,
    ):
        self.W = 1.85
        self.H = 4.084

        self.n_future = n_future
        self.reset()

    def reset(self):
        self.obj_col = torch.zeros(self.n_future)
        self.obj_box_col = torch.zeros(self.n_future)
        self.L2 = torch.zeros(self.n_future)
        self.total = torch.tensor(0)

    def evaluate_single_coll(self, traj, fut_boxes):
        n_future = traj.shape[0]
        yaw = get_yaw(traj)
        ego_box = traj.new_zeros((n_future, 7))
        ego_box[:, :2] = traj
        ego_box[:, 3:6] = ego_box.new_tensor([self.H, self.W, 1.56])
        ego_box[:, 6] = yaw
        collision = torch.zeros(n_future, dtype=torch.bool)

        for t in range(n_future):
            ego_box_t = ego_box[t].clone()
            boxes = fut_boxes[t][0].clone()
            collision[t] = check_collision(ego_box_t, boxes)
        return collision

    def evaluate_coll(self, trajs, gt_trajs, fut_boxes):
        B, n_future, _ = trajs.shape
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=trajs.device)
        obj_box_coll_sum = torch.zeros(n_future, device=trajs.device)

        assert B == 1, 'only supprt bs=1'
        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], fut_boxes)
            box_coll = self.evaluate_single_coll(trajs[i], fut_boxes)
            box_coll = torch.logical_and(box_coll, torch.logical_not(gt_box_coll))
            
            obj_coll_sum += gt_box_coll.long()
            min_v = box_coll.long().max()

            # box_coll = box_coll * 0 + min_v
            obj_box_coll_sum += box_coll.long()
        for i in range(n_future):
            obj_box_coll_sum[i] = obj_box_coll_sum[0:i+1].max()
        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        '''
        trajs: torch.Tensor (B, n_future, 3)
        gt_trajs: torch.Tensor (B, n_future, 3)
        '''
        return torch.sqrt((((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(dim=-1)) 

    def update(self, trajs, gt_trajs, gt_trajs_mask, fut_boxes):
        assert trajs.shape == gt_trajs.shape
        trajs[..., 0] = - trajs[..., 0]
        gt_trajs[..., 0] = - gt_trajs[..., 0]
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:, :, :2], gt_trajs[:, :, :2], fut_boxes)

        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(dim=0)
        self.total += len(trajs)

    def compute(self):
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'L2' : self.L2 / self.total
        }


def planning_eval(result_path, eval_config, logger):
    dataset = build_dataset(eval_config)
    dataloader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=8, shuffle=False, dist=False)
    planning_metrics = PlanningMetric()

    results = mmcv.load(result_path)

    for i, data in enumerate(tqdm(dataloader)):
        sdc_planning = data['gt_ego_fut_trajs'][0].data[0][0].cumsum(dim=-2).unsqueeze(0)
        sdc_planning_mask = data['gt_ego_fut_masks'][0].data[0][0].unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)
        command = data['gt_ego_fut_cmd'][0].data[0][0].argmax(dim=-1).item()
        fut_boxes = data['fut_boxes'][0]
        if not sdc_planning_mask.all():
            continue
        token = data['img_metas'][0].data[0][0]['token']
        res = results[i]['pts_plan_reg']['plan_reg']
        pred_sdc_traj = torch.tensor(res).to(sdc_planning).unsqueeze(0)
        # pred_sdc_traj[..., 0:1] += 0.985793 + 0.5
        # sdc_planning[..., 0:1] += 0.985793 + 0.5
        planning_metrics.update(pred_sdc_traj[:, :6, :2], sdc_planning[:, :6, :2], sdc_planning_mask[0, :, :6, :2], fut_boxes)
       
    planning_results = planning_metrics.compute()
    planning_metrics.reset()
    from prettytable import PrettyTable
    planning_tab = PrettyTable()
    metric_dict = {}

    planning_tab.field_names = [
    "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "avg"]
    for key in planning_results.keys():
        value = planning_results[key].tolist()
        new_values = []
        for i in range(len(value)):
            new_values.append(np.array(value[:i+1]).mean())
        value = new_values
        avg = [value[1], value[3], value[5]]
        avg = sum(avg) / len(avg)
        value.append(avg)
        metric_dict[key] = avg
        row_value = []
        row_value.append(key)
        for i in range(len(value)):
            if 'col' in key:
                row_value.append('%.3f' % float(value[i]*100) + '%')
            else:
                row_value.append('%.4f' % float(value[i]))
        planning_tab.add_row(row_value)

    print_log('\n'+str(planning_tab), logger=logger)
    return metric_dict



if __name__ == "__main__":
    point_cloud_range = [-54.0, -54.0, -3.0, 54.0, 54.0, 5.0]
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    map_class_names = [
        'ped_crossing', 'divider', 'boundary',
    ]
    occ_class_names = [
        'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk',
        'terrain', 'manmade', 'vegetation', 'free'
    ]

    seg_map_classes = [
        'drivable_area',
        'ped_crossing',
        'walkway',
        'stop_line',
        'carpark_area',
        'divider'
    ]

    out_size_factor = 2

    dataset_type = 'NuScenes3DDataset'
    data_root = 'data/nuscenes/'
    input_modality = dict(
        use_lidar=True,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False)

    voxel_size = [0.3, 0.3, 0.25]
    grid_size = [360, 360, 32]
    lion_dim = 128

    bda_aug_conf = dict(
        rot_lim=(-22.5 * 0, 22.5 * 0),
        scale_lim=(1.0, 1.0),
        flip_dx_ratio=0.0,
        flip_dy_ratio=0.0,
        tran_lim=[0.0, 0.0, 0.0]
    )

    grid_config = {
        'x': [-54.0, 54.0, 0.3],
        'y': [-54.0, 54.0, 0.3],
        'z': [-3, 5, 0.25],
        'depth': [1.0, 60.0, 0.5],
    }

    data_config = {
        'cams': [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ],
        'Ncams': 6,
        'input_size': (384, 1056),
        'src_size': (900, 1600),

        # Augmentation
        'resize': (-0.06, 0.11),
        'rot': (-5.4, 5.4),
        'flip': True,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.00,
    }

    tasks = dict(
        det=True,
        occ=False,
        map=False,
        motion=True,
        planning=True,
    )

    train_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
        ),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            use_dim=[0, 1, 2, 3, 4],
        ),
        dict(type='ToEgo'),
        dict(
            type='PrepareImageInputs',
            is_train=True,
            data_config=data_config),
        dict(type='LoadMultiTaskAnnotations3D'),
        dict(
            type='VectorizeMap',
            roi_size=(60, 30),
            simplify=False,
            normalize=False,
            sample_num=20,
            permute=True,
        ),
        dict(
            type='BEVAug',
            bda_aug_conf=bda_aug_conf,
            classes=class_names),
        dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='MultiObjectRangeFilter', point_cloud_range=point_cloud_range),
        dict(type='MultiObjectNameFilter', classes=class_names),
        dict(type='PointShuffle'),
        dict(type='MultiTaskFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['points',
                                    'img',
                                    'img_inputs',
                                    'gt_depth',
                                    'gt_bboxes_3d',
                                    'gt_labels_3d',
                                    'gt_occ',
                                    'mask_camera',
                                    'gt_map_labels',
                                    'gt_map_pts',
                                    'gt_bev_masks',
                                    'ego_status',
                                    'gt_agent_fut_trajs',
                                    'gt_agent_fut_masks',
                                    'gt_ego_fut_trajs',
                                    'gt_ego_fut_masks',
                                    'gt_ego_fut_cmd'],
            meta_keys=('filename', 'lidar2img'))
    ]
    test_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
        ),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            use_dim=[0, 1, 2, 3, 4],
        ),
        dict(type='ToEgo'),
        dict(
            type='PrepareImageInputs',
            is_train=False,
            data_config=data_config),
        dict(type='LoadMultiTaskAnnotations3D'),
        dict(
            type='VectorizeMap',
            roi_size=(60, 30),
            simplify=True,
            permute=False,
            normalize=False
        ),
        dict(
            type='BEVAug',
            bda_aug_conf=bda_aug_conf,
            classes=class_names),
        dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[0, 0],
                    scale_ratio_range=[1.0, 1.0],
                    translation_std=[0, 0, 0]),
                dict(type='RandomFlip3D'),
                dict(
                    type='MultiTaskFormatBundle3D',
                    class_names=class_names,
                    with_label=False),
                dict(type='Collect3D', keys=['points',
                                            'img',
                                            'img_inputs',
                                            'gt_depth',
                                            # 'vectors',
                                            'gt_bev_masks',
                                            'ego_status',
                                            'fut_boxes',
                                            'gt_ego_fut_trajs',
                                            'gt_ego_fut_masks',
                                            'gt_ego_fut_cmd'],
                    meta_keys=('filename', 'token'))
            ])
    ]

    data_basic_config = dict(
        type=dataset_type,
        data_root=data_root,
        occ_root='occ3d',
        classes=class_names,
        map_classes=map_class_names,
        seg_map_config=dict(
            seg_map_classes=seg_map_classes,
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5]
        ),
        bda_aug_conf=bda_aug_conf,
        occ_class=occ_class_names,
        modality=input_modality,
        version="v1.0-trainval",
        keep_consistent_seq_aug=True,
        sequences_split_num=-1
    )
    eval_config = dict(
        **data_basic_config,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
    )

    data = dict(
        samples_per_gpu=2,
        workers_per_gpu=4,
        train=dict(
            **data_basic_config,
            ann_file=data_root + '/nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            test_mode=False),
        val=dict(
            **data_basic_config,
            ann_file=data_root + '/nuscenes_infos_val.pkl',
            load_interval=1,
            pipeline=test_pipeline,
            test_mode=True,
            eval_config=eval_config),
        test=dict(
            **data_basic_config,
            ann_file=data_root + '/nuscenes_infos_val.pkl',
            load_interval=1,
            pipeline=test_pipeline,
            test_mode=True,
            eval_config=eval_config))

    optimizer = dict(
        type='AdamW',
        lr=0.0001,
        paramwise_cfg=dict(
            custom_keys={
                'img_backbone': dict(lr_mult=0.1)
            }),
        weight_decay=0.05
    )

    optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

    lr_config = dict(
        policy='cyclic',
        target_ratio=(10, 0.0001),
        cyclic_times=1,
        step_ratio_up=0.4)
    momentum_config = dict(
        policy='cyclic',
        target_ratio=(0.8947368421052632, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    total_epochs = 24
    checkpoint_config = dict(interval=1, max_keep_ckpts=12)
    log_config = dict(
        interval=50,
        hooks=[dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')])

    dist_params = dict(backend='nccl')
    log_level = 'INFO'
    work_dir = None
    load_from = 'ckpts/mm_lion_fusion_swin_384_perception.pth'
    resume_from = None
    workflow = [('train', 1)]
    gpu_ids = range(0, 8)

    eval_mode = dict(
        with_det=False,
        with_tracking=False,
        with_occ=tasks['occ'],
        with_map=tasks['map'],
        with_motion=tasks['motion'],
        with_planning=tasks['planning'],
        tracking_threshold=0.2,
        motion_threshhold=0.2,
    )
    evaluation = dict(
        interval=4,
        eval_mode=eval_mode,
    )

    result_path = 'ckpts/mm_lion_seq_fusion_swin_384_e2e_anchor_2ego/results.pkl'

    nusc = NuScenes(
        version='v1.0-trainval', dataroot=data_root, verbose=False)
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
    }
 
    planning_results_dict = planning_eval(result_path, eval_config, logger=None)
    results_dict.update(planning_results_dict)
    print(results_dict)