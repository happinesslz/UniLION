import copy
import os

import numpy as np
import mmcv
from mmcv import Config
from torch.utils.data import DataLoader
from mmdet.datasets import build_dataset, build_dataloader

from projects.mmdet3d_plugin.datasets.ego_pose_dataset import EgoPoseDataset
from projects.mmdet3d_plugin.datasets.evaluation.occ.miou_metrics import MetricIoU
from projects.mmdet3d_plugin.datasets.evaluation.occ.ray_metrics import main as calc_rayiou

INTERP_NUM = 200 # number of points to interpolate during evaluation
THRESHOLDS = [0.5, 1.0, 1.5] # AP thresholds
N_WORKERS = 16 # num workers to parallel


class OCCEvaluate:
    def __init__(self, data_infos, occ_root, occ_class_names) -> None:
        self.data_infos = data_infos
        self.occ_root = occ_root
        self.dataloader = DataLoader(EgoPoseDataset(self.data_infos), num_workers=8)
        self.iou_metric = MetricIoU()
        self.occ_class_names = occ_class_names

    def evaluate(self, result_path, metric='ray-iou', logger=None):
        occ_results = mmcv.load(result_path)

        occ_gts = []
        occ_preds = []
        occ_flow_gts = []
        occ_flow_preds = []
        lidar_origins = []
        inst_gts = []

        print('\nStarting Evaluation...')

        sample_tokens = [info['token'] for info in self.data_infos]

        for i, batch in enumerate(self.dataloader):
            token = batch[0][0]
            output_origin = batch[1]
            data_id = sample_tokens.index(token)

            info = self.data_infos[data_id]
            token = info['token']

            occ_path = os.path.join(self.occ_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path)
            gt_semantics = occ_gt['semantics']  # (Dx, Dy, Dz)
            occ_pred = occ_results[i]['pts_occ']['occ']  # (Dx, Dy, Dz)

            lidar_origins.append(copy.deepcopy(output_origin))
            occ_gts.append(copy.deepcopy(gt_semantics))
            occ_preds.append(copy.deepcopy(occ_pred))

            if 'flow' in occ_gt:
                gt_flow = occ_gt['flow']
                occ_pred_flow = occ_results[i]['pts_occ']['flow']
                occ_flow_gts.append(copy.deepcopy(gt_flow))
                occ_flow_preds.append(copy.deepcopy(occ_pred_flow))

            if 'mask_camera' in occ_gt:
                mask_lidar = occ_gt['mask_lidar'].astype(bool)  # (Dx, Dy, Dz)
                mask_camera = occ_gt['mask_camera'].astype(bool)  # (Dx, Dy, Dz)

                self.iou_metric.add_batch(
                    occ_pred,  # (Dx, Dy, Dz)
                    gt_semantics,  # (Dx, Dy, Dz)
                    mask_lidar,  # (Dx, Dy, Dz)
                    mask_camera  # (Dx, Dy, Dz)
                )

        if 'flow' in occ_gt:
            eval_results, table = calc_rayiou(occ_preds, occ_gts, lidar_origins, self.occ_class_names, occ_flow_preds, occ_flow_gts)
        else:
            eval_results, table = calc_rayiou(occ_preds, occ_gts, lidar_origins, self.occ_class_names)

        from mmcv.utils import print_log
        print_log('\n' + str(table), logger=logger)

        if 'mask_camera' in occ_gt:
            iou_eval_results, iou_table = self.iou_metric.count_miou()
            print_log('\n' + str(iou_table), logger=logger)
            eval_results.update(iou_eval_results)

        return eval_results
