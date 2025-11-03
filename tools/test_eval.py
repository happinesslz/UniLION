import os

import mmcv
from mmcv import Config
import argparse

from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import planning_eval
from projects.mmdet3d_plugin.datasets.evaluation.occ.occ_eval import OCCEvaluate


occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    # parser.add_argument('config', help='config file path')
    # parser.add_argument('--result-path',
    #                     default=None,
    #                     help='prediction result to visualize'
    #                          'If submission file is not provided, only gt will be visualized')
    # parser.add_argument(
    #     '--out-dir',
    #     default='vis',
    #     help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # cfg = Config.fromfile(args.config)
    # planning_eval(args.result_path, cfg.eval_config, logger=None)
    data_root = 'data/nuscenes'
    data = mmcv.load(data_root + '/nuscenes_infos_val.pkl', file_format="pkl")
    data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    occ_evaluator = OCCEvaluate(data_infos, os.path.join(data_root, 'openocc'), occ_class_names)
    res_path = '/data/zliu/projects/code/MM-LION/work_dirs/mm_lion_openocc/results.pkl'
    occ_results_dict = occ_evaluator.evaluate(res_path, logger=None)


if __name__ == '__main__':
    main()
