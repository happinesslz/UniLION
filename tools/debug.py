import pickle
import copy
import prettytable
from mmcv.utils import print_log
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory as det_configs
from projects.mmdet3d_plugin.datasets.evaluation.motion.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion


if __name__ == "__main__":
    results = pickle.load(open('/data/jhhou/disk/temp/motion.bin', 'rb'))
    output_dir = '/data/jhhou/disk/temp'
    det3d_eval_version="detection_cvpr_2019"
    version = "v1.0-trainval"
    data_root = 'data/nuscenes/'
    det3d_eval_configs = det_configs(det3d_eval_version)
    det3d_eval_configs.class_names = list(det3d_eval_configs.class_range.keys())

    nusc = NuScenes(
        version=version, dataroot=data_root, verbose=False)
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
    }
    nusc_eval = NuScenesEvalMotion(
        nusc,
        config=copy.deepcopy(det3d_eval_configs),
        result_path=results,
        eval_set=eval_set_map[version],
        output_dir='/data/jhhou/disk/temp',
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
    print_log('\n'+str(table))