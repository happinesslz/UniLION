import torch
import numpy as np
from mamba_ssm import Block as MambaBlock
from mmdet.core import multi_apply
from mmdet3d.models.builder import HEADS, build_loss
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from projects.mmdet3d_plugin.models.utils import TransformerDecoderLayer, gen_sineembed_for_position
import torch.utils.checkpoint as cp

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
class NaiveMotionHead(nn.Module):
    def __init__(
            self,
            num_classes=10,
            motion_anchor=None,
            in_channels=128 * 3,
            hidden_channel=128,
            num_trajs=6,
            # config for Transformer
            decoder_layer=dict(),
            num_heads=8,
            bn_momentum=0.1,
            bias='auto',
            # loss
            loss_motion_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
            loss_motion_cls=dict(),
            # others
            with_motion=False,
            train_cfg=None,
            test_cfg=None,
            with_cp=True,
    ):
        super(NaiveMotionHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_channel = hidden_channel
        self.bn_momentum = bn_momentum
        self.fut_steps = 6
        self.ego_fut_mode = 3
        self.fut_seconds = 6
        self.num_trajs = num_trajs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.motion = with_motion
        self.with_cp = with_cp

        self.loss_motion_reg = build_loss(loss_motion_reg)
        self.loss_motion_cls = build_loss(loss_motion_cls)

        self.fuser = ConvFuser(self.in_channels, self.hidden_channel)

        self.decoder = TransformerDecoderLayer(
            hidden_channel,
            num_heads,
            1024,
        )

        # Prediction Head
        agent_fut_decoder = list()
        for i in range(2):
            agent_fut_decoder.append(nn.Linear(self.hidden_channel, self.hidden_channel))
            agent_fut_decoder.append(nn.ReLU())

        self.reg = nn.Linear(self.hidden_channel, self.fut_seconds * 2 * 2)
        self.cls = nn.Linear(self.hidden_channel, 1)
        
        self.agent_fut_decoder = nn.Sequential(*agent_fut_decoder)

        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )

        self.motion_anchor_encoder = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(),
            nn.LayerNorm(hidden_channel),
            nn.Linear(hidden_channel, hidden_channel),
        )

        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

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

    def forward_single(self, feats, aux_feats, det_outs):
        batch_size = det_outs["heatmap"].shape[0]
        batch_score = det_outs["heatmap"].sigmoid()
        det_queries = det_outs['query_feat']

        one_hot = F.one_hot(
            det_outs['query_labels'], num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * det_outs["query_heatmap_score"] * one_hot
        cls_ids = batch_score.argmax(dim=1)
        motion_anchor = self.motion_anchor[cls_ids]
        motion_anchor = rearrange(motion_anchor[:, :, :, -1:], 'b l k n c -> b (l k) (n c)')
        motion_anchor = gen_sineembed_for_position(motion_anchor, self.hidden_channel)
        motion_anchor = self.motion_anchor_encoder(motion_anchor)
        motion_anchor = rearrange(motion_anchor, 'b l c -> b c l')
        det_queries = repeat(det_queries, 'b c l -> b c (l k)', k=self.num_trajs)
        det_queries = det_queries + motion_anchor

        feats = feats.permute(0, 1, 3, 2)
        aux_feats = aux_feats.permute(0, 1, 3, 2)
        feats = self.fuser(feats, aux_feats)

        bev_pos = self.bev_pos.repeat(feats.shape[0], 1, 1).to(feats.device)

        feats = feats.flatten(-2)
        if self.with_cp:
            agent_queries = cp.checkpoint(self.decoder, det_queries, feats, None, bev_pos, None, None)
        else:
            agent_queries = self.decoder(det_queries, feats, None, bev_pos)
        agent_queries = self.agent_fut_decoder(rearrange(agent_queries, 'b c l -> b l c'))

        outputs_agent_trajs = self.reg(agent_queries)
        outputs_agent_trajs_cls = self.cls(agent_queries)

        outputs_agent_trajs = rearrange(outputs_agent_trajs, 'b (l k) (n c) -> b l k n c', k=self.num_trajs, n=self.fut_seconds * 2)
        outputs_agent_trajs_cls = rearrange(outputs_agent_trajs_cls, 'b (l k) c -> b l (k c)', k=self.num_trajs)

        res = {'agent_fut_preds': outputs_agent_trajs, 'agent_fut_preds_cls': outputs_agent_trajs_cls, 'agent_queries': agent_queries}
        return [res]

    def forward(self, feats, map_feats, det_outs):
        if not isinstance(feats, list):
            feats = [feats]
        if not isinstance(map_feats, list):
            map_feats = [map_feats]
        res = multi_apply(self.forward_single, feats, map_feats, det_outs)
        assert len(res) == 1, "only support one level features."
        return res

    def get_best_reg(self, pred_trajs, gt_trajs):
        num_pred, mode, ts, d = pred_trajs.shape
        reg_preds_cum = pred_trajs.cumsum(dim=-2)
        reg_gt_cum = gt_trajs.cumsum(dim=-2)
        dist = torch.linalg.norm(reg_gt_cum.unsqueeze(1) - reg_preds_cum, dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        target_cls = mode_idx

        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, ts, d)
        best_reg = torch.gather(pred_trajs, 1, mode_idx).squeeze(1)
        return best_reg, target_cls

    def loss(self, preds_dicts, metas, assigned_agent_gt_inds, gt_agent_fut_trajs, gt_agent_fut_masks):
        loss = dict()

        agent_fut_preds = preds_dicts[0][0]["agent_fut_preds"]
        agent_fut_preds_cls = preds_dicts[0][0]["agent_fut_preds_cls"]

        agent_fut_preds = [agent_fut_preds[i, gt_inds[:, 0]] for i, gt_inds in enumerate(assigned_agent_gt_inds)]
        agent_fut_preds_cls = [agent_fut_preds_cls[i, gt_inds[:, 0]] for i, gt_inds in enumerate(assigned_agent_gt_inds)]
        gt_agent_fut_trajs = [gt_agent_fut_trajs[i][gt_inds[:, 1]] for i, gt_inds in enumerate(assigned_agent_gt_inds)]
        
        agent_fut_preds = torch.cat(agent_fut_preds)
        agent_fut_preds_cls = torch.cat(agent_fut_preds_cls)
        gt_agent_fut_trajs = torch.cat(gt_agent_fut_trajs)
        gt_agent_fut_masks = torch.cat(gt_agent_fut_masks).unsqueeze(-1).repeat(1, 1, 2)

        agent_fut_preds, cls_target = self.get_best_reg(agent_fut_preds, gt_agent_fut_trajs)

        agent_fut_preds = agent_fut_preds.cumsum(dim=-2)
        gt_agent_fut_trajs = gt_agent_fut_trajs.cumsum(dim=-2)

        loss_motion_reg = self.loss_motion_reg(
            agent_fut_preds,
            gt_agent_fut_trajs,
            gt_agent_fut_masks
        )

        loss_motion_cls = self.loss_motion_cls(agent_fut_preds_cls, cls_target)

        loss['motion_loss_reg'] = loss_motion_reg
        loss['loss_motion_cls'] = loss_motion_cls

        return loss