import prettytable
from typing import Dict, List, Optional
from time import time
from copy import deepcopy
from multiprocessing import Pool
from logging import Logger
from functools import partial, cached_property

import numpy as np
import torch
from numpy.typing import NDArray
from shapely.geometry import LineString

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader

from .AP import instance_match, average_precision

N_WORKERS = 16 # num workers to parallel


class SegEvaluate(object):
    """Evaluator for vectorized map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, dataset_cfg: Config, n_workers: int=N_WORKERS) -> None:
        self.dataset = build_dataset(dataset_cfg)
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=n_workers, shuffle=False, dist=False)
        self.map_classes = self.dataset.SEG_MAP_CLASSES
        
    def evaluate(self, 
                 result_path: str,
                 logger: Optional[Logger]=None) -> Dict[str, float]:

        results = mmcv.load(result_path)
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        pbar = mmcv.ProgressBar(len(results))
        for i, batch in enumerate(self.dataloader):
            pred = results[i]["pts_seg_map"]['seg']
            label = batch["gt_bev_masks"][0].data[0]

            pred = pred.reshape(num_classes, -1)
            label = label.bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

            pbar.update()

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        for index, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[index].max().item()
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f"map/{name}/iou@{threshold.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()

        from prettytable import PrettyTable
        table = prettytable.PrettyTable(['category'] +
                                        [f'IoU@{thr.item():.2f}' for thr in thresholds] + ['IoU@max'])
        for index, name in enumerate(self.map_classes):
            table.add_row([
                name,
                *[f'{iou.item():.4f}' for iou in ious[index]],
                f'{ious[index].max().item():.4f}',
            ])

        from mmcv.utils import print_log
        print_log('\n'+str(table), logger=logger)
        print_log(f'mIoU@max = {metrics["map/mean/iou@max"]:.4f}\n', logger=logger)


        return metrics