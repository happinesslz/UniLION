import copy


import torch
import numpy as np
from mamba_ssm import Block as MambaBlock
from mmdet.core import multi_apply
from mmdet3d.models.builder import HEADS, build_loss
import torch.nn as nn
from einops import repeat, rearrange

from projects.mmdet3d_plugin.models.utils import TransformerDecoderLayer, gen_sineembed_for_position
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class ConvFuser(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ConvFuser, self).__init__()
        self.fuser = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x1, x2):
        assert x2.size() == x2.size()
        # print('###x1:', x1.shape)
        # print('###x2:', x2.shape)
        x = torch.cat([x1, x2], dim=1)
        y = self.fuser(x)
        return y


@HEADS.register_module()
class AnchorPlannerHead(nn.Module):
    def __init__(
            self,
            planning_anchor=None,
            in_channels=128 * 3,
            hidden_channel=128,
            # config for Transformer
            num_decoder_layers=3,
            decoder_layer=dict(),
            num_heads=8,
            bn_momentum=0.1,
            bias='auto',
            # loss
            loss_plan_cls=dict(),
            loss_plan_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
            loss_ego_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
            # others
            train_cfg=None,
            test_cfg=None,
    ):
        super(AnchorPlannerHead, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.hidden_channel = hidden_channel
        self.bn_momentum = bn_momentum
        self.fut_steps = 6
        self.ego_fut_mode = 3
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_cls = build_loss(loss_plan_cls)
        self.loss_ego_reg = build_loss(loss_ego_reg)

        self.fuser = ConvFuser(self.in_channels, self.hidden_channel)

        decoder = list()
        for i in range(self.num_decoder_layers):
            decoder.append(TransformerDecoderLayer(
                                hidden_channel,
                                num_heads,
                                1024)
            )
        self.decoder = nn.ModuleList(decoder)

        planning_anchor = np.load(planning_anchor)
        self.planning_anchor = nn.Parameter(
            torch.tensor(planning_anchor, dtype=torch.float32),
            requires_grad=False,
        )

        self.planning_anchor_encoder = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.ego_encoder = nn.Sequential(
            nn.Linear(10, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        # Prediction Head
        ego_fut_decoder = list()
        for i in range(2):
            ego_fut_decoder.append(nn.Linear(self.hidden_channel, self.hidden_channel))
            ego_fut_decoder.append(nn.ReLU())
        
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        self.reg = nn.Linear(self.hidden_channel, self.fut_steps * 2)
        self.cls = nn.Linear(self.hidden_channel, 1)

        ego = list()
        for i in range(2):
            ego.append(nn.Linear(self.hidden_channel, self.hidden_channel))
            ego.append(nn.ReLU())
        ego.append(nn.Linear(self.hidden_channel, 10))
        self.ego = nn.Sequential(*ego)

        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.ego_info = MLN(3, f_dim=self.hidden_channel)

        kernel_size = tuple([int(x / 2) for x in [180, 180]])
        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channel),
            nn.Conv2d(self.hidden_channel, self.hidden_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channel),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size),
        )

        self.ego_anchor = nn.Parameter(
            torch.tensor([[0, 0.5, -1.84 + 1.56 / 2, np.log(4.08), np.log(1.73), np.log(1.56), 1, 0, 0, 0, 0],], dtype=torch.float32),
            requires_grad=False,
        )

        self.ego_anchor_encoder = nn.Sequential(
            nn.Linear(11, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel, bias=False),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.init_weights()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward_single(self, feats, aux_feats, ego_status, gt_ego_fut_cmd, gt_trajs, img_metas):
        feats = self.fuser(feats, aux_feats)
        bev_pos = self.bev_pos.repeat(feats.shape[0], 1, 1).to(feats.device)

        gt_ego_fut_cmd_feat = gt_ego_fut_cmd.to(feats.dtype).unsqueeze(-1)
        gt_ego_fut_cmd_feat = rearrange(gt_ego_fut_cmd_feat, 'b c l -> b l c')
        ego_anchor = self.ego_anchor.repeat(feats.shape[0], 1)

        ego_query = self.ego_feature_encoder(feats)
        ego_query = ego_query.unsqueeze(1).squeeze(-1).squeeze(-1)
        ego_query = self.query_encoder(ego_query)

        feats = feats.flatten(-2)
        for i in range(self.num_decoder_layers):
            ego_query = self.decoder[i](rearrange(ego_query, 'b l c -> b c l'), feats, None, bev_pos)
            ego_query = rearrange(ego_query, 'b c l -> b l c')
        
        ego_anchor_embed = self.ego_anchor_encoder(ego_anchor)
        ego_anchor_embed = ego_anchor_embed.unsqueeze(1)
        ego_status_query = ego_query + ego_anchor_embed
        outputs_ego_status = self.ego(ego_status_query)

        planning_anchor = self.planning_anchor.unsqueeze(0).repeat(len(feats), 1, 1, 1, 1)[:, :, :, -1:].flatten(-2)
        planning_anchor = gen_sineembed_for_position(planning_anchor, self.hidden_channel)
        planning_anchor = self.planning_anchor_encoder(planning_anchor)
        planning_anchor = rearrange(planning_anchor, 'b l k c -> b (l k) c')

        ego_status_feat = self.ego_encoder(outputs_ego_status.squeeze(1)).unsqueeze(1)
        ego_query = ego_query + planning_anchor + ego_status_feat
        gt_ego_fut_cmd_feat = repeat(gt_ego_fut_cmd_feat, 'b l c -> b (l k) c', k=ego_query.shape[1])
        ego_query = self.ego_info(ego_query, gt_ego_fut_cmd_feat)

        ego_query = self.ego_fut_decoder(ego_query)
        outputs_ego_trajs = self.reg(ego_query)
        outputs_ego_trajs_cls = self.cls(ego_query)

        outputs_ego_trajs = rearrange(outputs_ego_trajs, 'b (m k) (n c) -> b m k n c', k=self.planning_anchor.shape[1], m=self.ego_fut_mode, n=self.fut_steps)
        outputs_ego_trajs_cls = rearrange(outputs_ego_trajs_cls.flatten(-2), 'b (m k) -> b m k', k=self.planning_anchor.shape[1], m=self.ego_fut_mode)

        res = {'ego_fut_preds': outputs_ego_trajs, 'ego_fut_preds_cls': outputs_ego_trajs_cls, 'ego_status_preds': outputs_ego_status, 'ego_fut_cmd': gt_ego_fut_cmd}
        return [res]

    def forward(self, feats, map_feats, ego_status, gt_ego_fut_cmd, gt_trajs=None, img_metas=None):
        if not isinstance(feats, list):
            feats = [feats]
        if not isinstance(map_feats, list):
            map_feats = [map_feats]
        res = multi_apply(self.forward_single, feats, map_feats, [ego_status], [gt_ego_fut_cmd], [gt_trajs], [img_metas])
        assert len(res) == 1, "only support one level features."
        return res

    def get_best_reg(self, pred_trajs, pred_trajs_cls, gt_trajs, ego_fut_cmd, reg_weight):
        batch_size, num_pred, mode, ts, d = pred_trajs.shape
        bs_indices = torch.arange(batch_size, device=pred_trajs.device)
        cmd = ego_fut_cmd.argmax(dim=-1)
        pred_trajs = pred_trajs[bs_indices, cmd].unsqueeze(1)
        best_cls = pred_trajs_cls[bs_indices, cmd].unsqueeze(1)

        reg_preds_cum = pred_trajs.cumsum(dim=-2)
        reg_gt_cum = gt_trajs.cumsum(dim=-2)
        dist = torch.linalg.norm(reg_gt_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
        dist = dist * reg_weight
      
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        target_cls = mode_idx

        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
        best_reg = torch.gather(pred_trajs, 2, mode_idx).squeeze(1)
        return best_cls, best_reg, target_cls

    def loss(self, preds_dicts, metas, gt_ego_fut_trajs, gt_ego_fut_masks, gt_ego_fut_cmd, gt_ego_status):
        ego_fut_preds = preds_dicts[0][0]["ego_fut_preds"]
        ego_fut_cmd = preds_dicts[0][0]["ego_fut_cmd"]
        ego_fut_preds_cls = preds_dicts[0][0]["ego_fut_preds_cls"] 
        ego_status_preds = preds_dicts[0][0]['ego_status_preds']
   
        gt_ego_fut_trajs = gt_ego_fut_trajs.unsqueeze(1)

        loss_plan_l1_weight = gt_ego_fut_masks[:, None, None, :]

        cls_pred, reg_pred, cls_target = self.get_best_reg(ego_fut_preds, ego_fut_preds_cls, gt_ego_fut_trajs, ego_fut_cmd, loss_plan_l1_weight)
        loss_plan_reg = self.loss_plan_reg(
            reg_pred.reshape(-1, self.fut_steps * 2),
            gt_ego_fut_trajs.reshape(-1, self.fut_steps * 2),
            loss_plan_l1_weight.repeat(1, 1, 1, reg_pred.shape[-1]).reshape(-1, self.fut_steps * 2)
        )

        loss_plan_cls_weight = loss_plan_l1_weight.squeeze(-1).any(dim=-1)
        loss_plan_cls = self.loss_plan_cls(cls_pred.flatten(end_dim=1), cls_target.flatten(end_dim=1), loss_plan_cls_weight.flatten(end_dim=1))

        loss_ego_reg = self.loss_ego_reg(ego_status_preds.reshape(-1, 10), gt_ego_status.reshape(-1, 10))

        loss = dict()

        loss['planning_loss_plan_reg'] = loss_plan_reg
        loss['planning_loss_plan_cls'] = loss_plan_cls
        loss['planning_loss_ego_reg'] = loss_ego_reg
   
        return loss

    def get_bboxes(self, preds_dicts, metas, results_det=None, **data):
        ego_fut_preds = preds_dicts[0][0]['ego_fut_preds']
        ego_fut_cmd = preds_dicts[0][0]['ego_fut_cmd']
        ego_fut_preds_cls = preds_dicts[0][0]["ego_fut_preds_cls"].sigmoid()
        select = ego_fut_cmd.argmax(dim=-1)

        gt_ego_fut_trajs = data['gt_ego_fut_trajs'].unsqueeze(1)
        loss_plan_l1_weight = data['gt_ego_fut_masks'][:, None, None, :]
        
        ret_layer = []
        for i in range(len(ego_fut_preds)):
            plan_select = select[i]
            plan_reg = ego_fut_preds[i].cumsum(dim=-2)
            plan_cls = ego_fut_preds_cls[i][plan_select]
            mode_idx = plan_cls.argmax(dim=-1)
            ret_layer.append(plan_reg[plan_select][mode_idx])

        assert len(ret_layer) == 1

        return ret_layer

    def get_yaw(self, traj, start_yaw=0, static_dis_thresh=0.5):
        yaw = traj.new_zeros(traj.shape[:-1])
        yaw[..., 1:-1] = torch.atan2(
            traj[..., 2:, 1] - traj[..., :-2, 1],
            traj[..., 2:, 0] - traj[..., :-2, 0],
        )
        yaw[..., -1] = torch.atan2(
            traj[..., -1, 1] - traj[..., -2, 1],
            traj[..., -1, 0] - traj[..., -2, 0],
        )
        yaw[..., 0] = start_yaw
        # for static object, estimated future yaw would be unstable
        start = traj[..., 0, :]
        end = traj[..., -1, :]
        dist = torch.linalg.norm(end - start, dim=-1)
        mask = dist < static_dis_thresh
        start_yaw = yaw[..., 0].unsqueeze(-1)
        yaw = torch.where(
            mask.unsqueeze(-1),
            start_yaw,
            yaw,
        )
        return yaw.unsqueeze(-1)


class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out
