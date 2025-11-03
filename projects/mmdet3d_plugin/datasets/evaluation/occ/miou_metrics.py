import numpy as np
from prettytable import PrettyTable


class MetricIoU:
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation', 'free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label, (N_valid, )
            gt (1-d array): gt_occupancu_label, (N_valid, )

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)  # N_total
        correct = np.sum((pred[k] == gt[k]))  # N_correct

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),  # (N_cls, N_cls),
            correct,  # N_correct
            labeled,  # N_total
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        """
        Args:
            pred: (N_valid, )
            label: (N_valid, )
            n_classes: int=18

        Returns:

        """
        hist = np.zeros((n_classes, n_classes))  # (N_cls, N_cls)
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist  # (N_cls, N_cls)
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        """
        Args:
            semantics_pred: (Dx, Dy, Dz, n_cls)
            semantics_gt: (Dx, Dy, Dz)
            mask_lidar: (Dx, Dy, Dz)
            mask_camera: (Dx, Dy, Dz)

        Returns:

        """
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]  # (N_valid, )
            masked_semantics_pred = semantics_pred[mask_camera]  # (N_valid, )
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist  # (N_cls, N_cls)  列对应每个gt类别，行对应每个预测类别, 这样只有对角线位置上的预测是准确的.

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)

        eval_res = dict()

        table = PrettyTable(['Class Names', 'IoU'])
        for ind_class in range(self.num_classes - 1):
            table.add_row([self.class_names[ind_class], f'{mIoU[ind_class]:.4f}'])
            eval_res[self.class_names[ind_class]] = mIoU[ind_class]
        table.add_row(['mean', f'{np.nanmean(mIoU[:self.num_classes - 1]):.4f}'])
        eval_res['mIoU'] = np.nanmean(mIoU[:self.num_classes - 1])

        return eval_res, table
