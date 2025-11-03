import os
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
from pyquaternion import Quaternion

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
 
CMD_LIST = ['Turn Left', 'Turn Right', 'Go Straight']
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']
SCORE_THRESH = 0.3
MAP_SCORE_THRESH = 0.3
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
color_mapping = np.asarray([
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
]) / 255

MAP_PALETTE = np.asarray([
    [166, 206, 227],
    [251, 154, 153],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [106, 61, 154],
])

occ_color_map = np.array([
    [0, 0, 0, 255],    # others
    [255, 120, 50, 255],  # barrier              orangey
    [255, 192, 203, 255],  # bicycle              pink
    [255, 255, 0, 255],  # bus                  yellow
    [0, 150, 245, 255],  # car                  blue
    [0, 255, 255, 255],  # construction_vehicle cyan
    [200, 180, 0, 255],  # motorcycle           dark orange
    [255, 0, 0, 255],  # pedestrian           red
    [255, 240, 150, 255],  # traffic_cone         light yellow
    [135, 60, 0, 255],  # trailer              brown
    [160, 32, 240, 255],  # truck                purple
    [255, 0, 255, 255],  # driveable_surface    dark pink
    [175,   0,  75, 255],       # other_flat           dark red
    [75, 0, 75, 255],  # sidewalk             dard purple
    [150, 240, 80, 255],  # terrain              light green
    [230, 230, 250, 255],  # manmade              white
    [0, 175, 0, 255],  # vegetation           green
    [255, 255, 255, 255],  # free             white
], dtype=np.uint8)

class BEVRender:
    def __init__(
        self, 
        plot_choices,
        out_dir,
        xlim = 40,
        ylim = 40,
    ):
        self.plot_choices = plot_choices
        self.xlim = xlim
        self.ylim = ylim
        self.gt_dir = os.path.join(out_dir, "bev_gt")
        self.pred_dir = os.path.join(out_dir, "bev_pred")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

    def reset_canvas(self):
        plt.close()
        self.fig, self.axes = plt.subplots(1, 1, figsize=(20, 20))
        if not (self.plot_choices['map'] or self.plot_choices['occ']):
            self.axes.set_xlim(- self.xlim, self.xlim)
            self.axes.set_ylim(- self.ylim, self.ylim)
        self.axes.axis('off')

    def render(
        self,
        data, 
        result,
        index,
    ):
        self.reset_canvas()

        if self.plot_choices['det'] or self.plot_choices['planning'] or self.plot_choices['motion']:
            self.draw_point(data)
            self._render_sdc_car()
        if self.plot_choices['det']:
            self.draw_detection_gt(data)
        if self.plot_choices['motion']:
            self.draw_motion_gt(data)
        if self.plot_choices['occ']:
            self.draw_occ_gt(data)
        if self.plot_choices['map']:
            self.draw_map_gt(data)
        if self.plot_choices['planning']:
            self.draw_planning_gt(data)
            # self._render_command(data)
            # self._render_legend()

        save_path_gt = os.path.join(self.gt_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_gt)
        self.reset_canvas()
   
        if self.plot_choices['det'] or self.plot_choices['planning'] or self.plot_choices['motion']:
            self.draw_point(data)
            self._render_sdc_car()
        if self.plot_choices['det']:
            self.draw_detection_pred(result['pts_bbox'])
        # self.draw_track_pred(result)
        if self.plot_choices['motion']:
            self.draw_motion_pred(result['pts_bbox'])
        if self.plot_choices['map']:
            self.draw_map_pred(result)
        if self.plot_choices['occ']:
            self.draw_occ_pred(result)
        if self.plot_choices['planning']:
            self.draw_planning_pred(data, result['pts_plan_reg'])
            # self._render_command(data)
            # self._render_legend()

        save_path_pred = os.path.join(self.pred_dir, str(index).zfill(4) + '.jpg')
        self.save_fig(save_path_pred)

        return save_path_gt, save_path_pred

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(filename)

    def draw_point(self, data):
        lidar_path = data['pts_filename']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        lidar2lidarego = np.eye(4, dtype=np.float32)
        lidar2lidarego[:3, :3] = Quaternion(
            data['lidar2ego_rotation']).rotation_matrix
        lidar2lidarego[:3, 3] = data['lidar2ego_translation']

        lidarego2global = np.eye(4, dtype=np.float32)
        lidarego2global[:3, :3] = Quaternion(
            data['ego2global_rotation']).rotation_matrix
        lidarego2global[:3, 3] = data['ego2global_translation']

        camego2global = np.eye(4, dtype=np.float32)
        camego2global[:3, :3] = Quaternion(
            data['cams']['CAM_FRONT']
            ['ego2global_rotation']).rotation_matrix
        camego2global[:3, 3] = data['cams']['CAM_FRONT'][
            'ego2global_translation']
        lidar2camego = np.linalg.inv(camego2global) @ lidarego2global @ lidar2lidarego

        points_ego = lidar2camego[:3, :3].reshape(1, 3, 3) @ \
                     points[:, :3].reshape(-1, 3, 1) + \
                     lidar2camego[:3, 3].reshape(1, 3, 1)
        points[:, :3] = points_ego.squeeze(-1)
        
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        norm = Normalize(vmin=min(distances), vmax=max(distances))

        self.axes.scatter(
            points[:, 0],       
            points[:, 1],          
            s=1,                 
            alpha=0.5,            
            c=distances, cmap='jet', norm=norm
        )

    def draw_detection_gt(self, data):
        for i in range(data['ann_info']['gt_labels_3d'].shape[0]):
            label = data['ann_info']['gt_labels_3d'][i]
            if label == -1: 
                continue
            # color = color_mapping[i % len(color_mapping)]
            color = np.array([0, 255, 0]) / 255

            # draw corners
            corners = box3d_to_corners(data['ann_info']['gt_bboxes_3d'].tensor)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_occ_gt(self, data):
        occ_path = data['ann_info']['occ_path']
        gt_occ = np.load(occ_path)
        gt_occ = gt_occ['semantics']

        H, W, D = gt_occ.shape

        free_id = len(occ_class_names) - 1
        semantics_2d = np.ones([H, W], dtype=np.int32) * free_id

        for i in range(D):
            semantics_i = gt_occ[..., i]
            non_free_mask = (semantics_i != free_id)
            semantics_2d[non_free_mask] = semantics_i[non_free_mask]

        viz = occ_color_map[semantics_2d]
        viz = viz[..., :3]
        self.axes.imshow(viz.transpose(1, 0, 2)[::-1, :, :])

    def draw_detection_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['det'] and "boxes_3d" in result):
            return

        bboxes = torch.cat([result['boxes_3d'].gravity_center, result['boxes_3d'].tensor[..., 3:]], dim=-1)
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['labels_3d'][i] % len(color_mapping)]
            color = np.array([0, 0, 255]) / 255

            # draw corners
            corners = box3d_to_corners(bboxes)[i, [0, 3, 7, 4, 0]]
            x = corners[:, 0]
            y = corners[:, 1]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

            # draw line to indicate forward direction
            forward_center = np.mean(corners[2:4], axis=0)
            center = np.mean(corners[0:4], axis=0)
            x = [forward_center[0], center[0]]
            y = [forward_center[1], center[1]]
            self.axes.plot(x, y, color=color, linewidth=3, linestyle='-')

    def draw_track_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['track'] and "anchor_queue" in result):
            return
        
        temp_bboxes = result["anchor_queue"]
        period = result["period"]
        bboxes = result['boxes_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]
            if score < SCORE_THRESH: 
                continue
            color = color_mapping[result['labels_3d'][i] % len(color_mapping)]
            center = bboxes[i, :3]
            centers = [center]
            for j in range(period[i]):
                # draw corners
                corners = box3d_to_corners(temp_bboxes[:, -1-j])[i, [0, 3, 7, 4, 0]]
                x = corners[:, 0]
                y = corners[:, 1]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')

                # draw line to indicate forward direction
                forward_center = np.mean(corners[2:4], axis=0)
                center = np.mean(corners[0:4], axis=0)
                x = [forward_center[0], center[0]]
                y = [forward_center[1], center[1]]
                self.axes.plot(x, y, color=color, linewidth=2, linestyle='-')
                centers.append(center)

            centers = np.stack(centers)
            xs = centers[:, 0]
            ys = centers[:, 1]
            self.axes.plot(xs, ys, color=color, linewidth=2, linestyle='-')

    def draw_motion_gt(self, data):
        if not self.plot_choices['motion']:
            return

        for i in range(data['ann_info']['gt_labels_3d'].shape[0]):
            label = data['ann_info']['gt_labels_3d'][i]
            if label == -1: 
                continue
            color = color_mapping[i % len(color_mapping)]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            center = data['ann_info']['gt_bboxes_3d'].tensor[i, :2]
            masks = data['ann_info']['gt_agent_fut_masks'][i].astype(bool)
            if masks[0] == 0:
                continue
            trajs = data['ann_info']['gt_agent_fut_trajs'][i][masks]
            trajs = trajs.cumsum(axis=0) + center.numpy()
            trajs = np.concatenate([center.reshape(1, 2), trajs], axis=0)

            self._render_traj(trajs, traj_score=1.0,
                            colormap='winter', dot_size=dot_size)

    def draw_motion_pred(self, result, top_k=3):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['motion'] and "trajs_3d" in result):
            return
        
        bboxes = result['boxes_3d'].tensor
        labels = result['labels_3d']
        for i in range(result['labels_3d'].shape[0]):
            score = result['scores_3d'][i]

            if score < SCORE_THRESH: 
                continue
            label = labels[i]
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25

            traj = result['trajs_3d'][i].numpy()
            traj_cls = result['trajs_cls_3d'][i].sigmoid()
            for j in range(len(traj)):
                if traj_cls[j] < SCORE_THRESH:
                    continue
                center = bboxes[i, :2][None].repeat(1, 1).numpy()
                traj_j = np.concatenate([center, traj[j]], axis=0)

                self._render_traj(traj_j, traj_score=score.item(),
                                colormap='winter', dot_size=dot_size)

    
    # def draw_map_gt(self, data):
    #     vectors = data['map_infos']
    #     for label, vector_list in vectors.items():
    #         color = COLOR_VECTORS[label]
    #         for vector in vector_list:
    #             pts = vector[:, :2]
    #             x = np.array([pt[0] for pt in pts])
    #             y = np.array([pt[1] for pt in pts])
    #             self.axes.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_occ_pred(self, result):
        pred_occ = result['pts_occ']['occ']
        pred_occ = np.array(pred_occ)

        H, W, D = pred_occ.shape

        free_id = len(occ_class_names) - 1
        semantics_2d = np.ones([H, W], dtype=np.int32) * free_id

        for i in range(D):
            semantics_i = pred_occ[..., i]
            non_free_mask = (semantics_i != free_id)
            semantics_2d[non_free_mask] = semantics_i[non_free_mask]

        viz = occ_color_map[semantics_2d]
        viz = viz[..., :3]
        self.axes.imshow(viz.transpose(1, 0, 2)[::-1, :, :])

    def draw_map_gt(self, data):
        gt_bev_masks = data['ann_info']['gt_bev_masks']
        gt_bev_masks = np.array(gt_bev_masks)
        
        canvas = np.zeros((*gt_bev_masks.shape[-2:], 3), dtype=np.uint8)
        canvas[:] = (255, 255, 255)
        for i in range(len(gt_bev_masks)):
            canvas[gt_bev_masks[i].astype(np.bool), :] = MAP_PALETTE[i]
        canvas = canvas[::-1, :, :]
        self.axes.imshow(canvas)

    # def draw_map_pred(self, result):
    #     if not (self.plot_choices['draw_pred'] and self.plot_choices['map'] and "vectors" in result):
    #         return

    #     for i in range(len(result['scores'])):
    #         score = result['scores'][i]
    #         if score < MAP_SCORE_THRESH:
    #             continue
    #         color = COLOR_VECTORS[result['labels'][i]]
    #         pts = np.array(result['vectors'][i])
    #         x = pts[:, 0]
    #         y = pts[:, 1]
    #         plt.plot(x, y, color=color, linewidth=3, marker='o', linestyle='-', markersize=7)

    def draw_map_pred(self, result):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['map'] and "pts_seg_map" in result):
            return

        map_pred = result['pts_seg_map']['seg']
        map_pred = np.array(map_pred)

        canvas = np.zeros((*map_pred.shape[-2:], 3), dtype=np.uint8)
        canvas[:] = (255, 255, 255)
        for i in range(len(map_pred)):
            canvas[(map_pred[i] > 0.2).astype(np.bool), :] = MAP_PALETTE[i]
        canvas = canvas[::-1, :, :]
        self.axes.imshow(canvas)


    def draw_planning_gt(self, data):
        # if not self.plot_choices['planning']:
        #     return

        # draw planning gt
        masks = data['ann_info']['gt_ego_fut_masks'].astype(bool)
        if masks[0] != 0:
            plan_traj = data['ann_info']['gt_ego_fut_trajs'][masks]
            cmd = data['gt_ego_fut_cmd']
            plan_traj[abs(plan_traj) < 0.01] = 0.0
            plan_traj = plan_traj.cumsum(axis=0)
            plan_traj = np.concatenate((np.zeros((1, plan_traj.shape[1])), plan_traj), axis=0)
            self._render_traj(plan_traj, traj_score=1.0,
                colormap='autumn', dot_size=50)

    def draw_planning_pred(self, data, result, top_k=3):
        if not (self.plot_choices['draw_pred'] and self.plot_choices['planning']):
            return

        plan_trajs = np.array(result['plan_reg'])
        plan_trajs = np.concatenate((np.zeros((1, 2)), plan_trajs), axis=0)
        viz_traj = plan_trajs
        self._render_traj(viz_traj, traj_score=1, colormap='autumn', dot_size=50)


    def _render_traj(
        self, 
        future_traj, 
        traj_score=1, 
        colormap='winter', 
        points_per_step=20, 
        dot_size=25
    ):
        total_steps = (len(future_traj) - 1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors * traj_score + \
            (1 - traj_score) * np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps - 1):
            unit_vec = future_traj[i // points_per_step +
                                   1] - future_traj[i // points_per_step]
            total_xy[i] = (i / points_per_step - i // points_per_step) * \
                unit_vec + future_traj[i // points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_sdc_car(self):
        sdc_car_png = cv2.imread('resources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        im = self.axes.imshow(sdc_car_png.transpose(1, 0, 2), extent=(2, -2, -1, 1))
        im.set_zorder(1)
        # W = 1.85
        # H = 4.084
        # x = [H / 2 + 0.5 + 0.985793, H / 2 + 0.5 + 0.985793, -H / 2 + 0.5 + 0.985793, -H / 2 + 0.5 + 0.985793, H / 2 + 0.5 + 0.985793]
        # y = [W / 2, -W / 2, -W / 2, W / 2, W / 2]
        # self.axes.plot(x, y, color=[0, 0, 0], linewidth=3, linestyle='-')

    def _render_legend(self):
        legend = cv2.imread('resources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(15, 40, -40, -30))

    def _render_command(self, data):
        cmd = data['gt_ego_fut_cmd'].argmax()
        self.axes.text(-38, -38, CMD_LIST[cmd], fontsize=60)