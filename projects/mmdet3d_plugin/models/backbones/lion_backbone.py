import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_scatter
import numpy as np
import spconv.pytorch as spconv
from mamba_ssm import Block as MambaBlock
from projects.mmdet3d_plugin.models.ops import RWKVBlock
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class Lion3DBackbone(nn.Module):
    def __init__(self,
                 grid_size,
                 num_layers,
                 depths,
                 layer_down_scales,
                 direction,
                 diffusion,
                 shift,
                 diff_scale,
                 window_shape,
                 group_size,
                 layer_dim,
                 linear_operator,
                 sfd=True,
                 **kwargs):
        super().__init__()

        self.sparse_shape = grid_size[::-1]
        norm_fn = partial(nn.LayerNorm)

        self.window_shape = window_shape
        self.group_size = group_size
        self.layer_dim = layer_dim
        self.linear_operator = linear_operator

        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)

        self.input_merge = VoxelMerging(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 1],
                                 norm_layer=norm_fn, diffusion=False, diff_scale=0, conv=False)

        self.dow0 = VoxelMerging(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 1],
                                 norm_layer=norm_fn, diffusion=False, diff_scale=0, conv=sfd)

        self.block1 = LionBlock(self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
                                self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0,
                                n_layer=self.n_layer, sfd=sfd)

        self.dow1 = VoxelMerging(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
                                 norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale, conv=sfd)

        self.block2 = LionBlock(self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
                                self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8,
                                n_layer=self.n_layer, sfd=sfd)

        self.dow2 = VoxelMerging(self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
                                  norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale, conv=sfd)

        self.block3 = LionBlock(self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
                                self.group_size[2], direction, shift=shift, operator=self.linear_operator,
                                layer_id=16, n_layer=self.n_layer, sfd=sfd)

        self.dow3 = VoxelMerging(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                 norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale, conv=sfd)

        self.block4 = LionBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                self.group_size[3], direction, shift=shift, operator=self.linear_operator,
                                layer_id=24, n_layer=self.n_layer, sfd=sfd)

        self.dow4 = VoxelMerging(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                 norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale, conv=sfd)

        self.layer_out = LionLayer(self.layer_dim[3], [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
                                   operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)

    def forward(self, voxel_features, voxel_coords, batch_size):
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x, _ = self.input_merge(x)
        x, _ = self.dow0(x)
        x = self.block1(x)
        x, _ = self.dow1(x)
        x = self.block2(x)
        x, _ = self.dow2(x)
        x = self.block3(x)
        x, _ = self.dow3(x)
        x = self.block4(x)
        x, _ = self.dow4(x)
        x = self.layer_out(x)

        return x


@torch.inference_mode()
def get_window_coors_shift_v2(coords, sparse_shape, window_shape, shift=False):
    sparse_shape_z, sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x, win_shape_y, win_shape_z = window_shape

    if shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.

    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    x = coords[:, 3] + shift_x
    y = coords[:, 2] + shift_y
    z = coords[:, 1] + shift_z

    win_coors_x = x // win_shape_x
    win_coors_y = y // win_shape_y
    win_coors_z = z // win_shape_z

    coors_in_win_x = x % win_shape_x
    coors_in_win_y = y % win_shape_y
    coors_in_win_z = z % win_shape_z

    batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + \
                       win_coors_y * max_num_win_z + win_coors_z
    batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z + \
                       win_coors_x * max_num_win_z + win_coors_z

    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds_x, batch_win_inds_y, coors_in_win


class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            group_size,
            shift,
    ) -> None:
        super().__init__()
        self.window_shape = window_shape
        self.group_size = group_size
        self.shift = shift

    def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        coords = coords.long()
        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
                torch.div(
                    batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                    self.group_size,
                    rounding_mode="trunc",
                )
                * self.group_size
        )

        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)

        for i in range(batch_size):
            if num_per_batch[i] != num_per_batch_p[i]:
                bias_index = batch_start_indices_p[i] - batch_start_indices[i]
                flat2win[
                batch_start_indices_p[i + 1] - self.group_size + (num_per_batch[i] % self.group_size):
                batch_start_indices_p[i + 1]
                ] = flat2win[
                    batch_start_indices_p[i + 1]
                    - 2 * self.group_size
                    + (num_per_batch[i] % self.group_size): batch_start_indices_p[i + 1] - self.group_size
                    ] if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
                    win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
                        (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
                    : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index

            win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

            flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}

        batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                     self.window_shape, self.shift)
        vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
        vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
              self.window_shape[2] + coors_in_win[..., 0]

        vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
        vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
              self.window_shape[2] + coors_in_win[..., 0]

        _, mappings["x"] = torch.sort(vx)
        _, mappings["y"] = torch.sort(vy)

        return mappings


class VoxelMerging(nn.Module):
    def __init__(self, dim, out_dim=-1, down_scale=[2, 2, 2], norm_layer=nn.LayerNorm, diffusion=False, diff_scale=0.2, conv=True):
        super().__init__()
        self.dim = dim
        self.conv = conv

        if out_dim == -1:
            out_dim = dim

        if conv:
            self.sub_conv = spconv.SparseSequential(
                spconv.SubMConv3d(dim, out_dim, 3, bias=False, indice_key='subm'),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )

        self.norm = norm_layer(out_dim)

        self.sigmoid = nn.Sigmoid()
        self.down_scale = down_scale
        self.diffusion = diffusion
        self.diff_scale = diff_scale

    def forward(self, x, coords_shift=1, diffusion_scale=4):

        if self.conv:
            x = self.sub_conv(x)

        d, h, w = x.spatial_shape
        down_scale = self.down_scale

        if self.diffusion:
            x_feat_att = x.features.mean(-1)
            batch_size = x.indices[:, 0].max() + 1
            selected_diffusion_feats_list = [x.features.clone()]
            selected_diffusion_coords_list = [x.indices.clone()]
            for i in range(batch_size):
                mask = x.indices[:, 0] == i
                valid_num = mask.sum()
                K = int(valid_num * self.diff_scale)
                _, indices = torch.topk(x_feat_att[mask], K)

                selected_coords_copy = x.indices[mask][indices].clone()
                selected_coords_num = selected_coords_copy.shape[0]
                selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1)
                selected_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0

                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (
                        selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (
                        selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 3:4] = (
                        selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 2:3] = (
                        selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num:selected_coords_num * 2, 1:2] = (
                    selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                if diffusion_scale == 4:
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (
                            selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (
                            selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (
                            selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (
                            selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (
                        selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_diffusion_coords_list.append(selected_coords_expand)
                selected_diffusion_feats_list.append(selected_feats_expand)

            coords = torch.cat(selected_diffusion_coords_list)
            final_diffusion_feats = torch.cat(selected_diffusion_feats_list)

        else:
            coords = x.indices.clone()
            final_diffusion_feats = x.features.clone()

        coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
        coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
        coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

        scale_xyz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1]) * (
                x.spatial_shape[2] // down_scale[0])
        scale_yz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1])
        scale_z = (x.spatial_shape[0] // down_scale[2])

        merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        features_expand = final_diffusion_feats

        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

        x_merge = torch_scatter.scatter_add(features_expand, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        x_merge = self.norm(x_merge)

        x_merge = spconv.SparseConvTensor(
            features=x_merge,
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        return x_merge, unq_inv


class VoxelExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, up_x, unq_inv):
        # z, y, x
        n, c = x.features.shape

        x_copy = torch.gather(x.features, 0, unq_inv.unsqueeze(1).repeat(1, c))
        up_x = replace_feature(up_x, up_x.features + x_copy)
        return up_x


LinearOperatorMap = {
    'Mamba': MambaBlock,
    'RWKV': RWKVBlock
}


class LionLayer(nn.Module):
    def __init__(self, dim, window_shape, group_size, direction, shift, operator=None, layer_id=0, n_layer=0):
        super(LionLayer, self).__init__()

        self.window_shape = window_shape
        self.group_size = group_size
        self.dim = dim
        self.direction = direction

        operator_cfg = operator.cfg
        operator_cfg['d_model'] = dim

        block_list = []
        for i in range(len(direction)):
            operator_cfg['layer_id'] = i + layer_id
            operator_cfg['n_layer'] = n_layer
            operator_cfg['with_cp'] = layer_id >= 0
            block_list.append(LinearOperatorMap[operator.type](**operator_cfg))

        self.blocks = nn.ModuleList(block_list)
        self.window_partition = FlattenedWindowMapping(self.window_shape, self.group_size, shift)

    def forward(self, x):
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)

        for i, block in enumerate(self.blocks):
            indices = mappings[self.direction[i]]
            x_features = x.features[indices][mappings["flat2win"]]
            x_features = x_features.view(-1, self.group_size, x.features.shape[-1])

            x_features = block(x_features)

            x.features[indices] = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]

        return x


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class LionBlock(nn.Module):
    def __init__(self, dim: int, depth: int, down_scales: list, window_shape, group_size, direction, shift=False,
                 operator=None, layer_id=0, n_layer=0, sfd=True):
        super().__init__()

        self.down_scales = down_scales

        self.encoder = nn.ModuleList()
        self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()

        norm_fn = partial(nn.LayerNorm)

        shift = [False, shift]
        for idx in range(depth):
            self.encoder.append(
                LionLayer(dim, window_shape, group_size, direction, shift[idx], operator, layer_id + idx * 2,
                          n_layer))
            self.pos_emb_list.append(PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim))
            self.downsample_list.append(VoxelMerging(dim, dim, down_scale=down_scales[idx], norm_layer=norm_fn, conv=sfd))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for idx in range(depth):
            self.decoder.append(LionLayer(dim, window_shape, group_size, direction, shift[idx], operator,
                                          layer_id + 2 * (idx + depth), n_layer))
            self.decoder_norm.append(norm_fn(dim))
            self.upsample_list.append(VoxelExpanding(dim))

    def forward(self, x):
        features = []
        index = []

        for idx, enc in enumerate(self.encoder):
            pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],
                                         embed_layer=self.pos_emb_list[idx])

            x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
            x = enc(x)
            features.append(x)
            x, unq_inv = self.downsample_list[idx](x)
            index.append(unq_inv)

        i = 0
        for dec, norm, up_x, unq_inv, up_scale in zip(self.decoder, self.decoder_norm, features[::-1],
                                                      index[::-1], self.down_scales[::-1]):
            x = dec(x)
            x = self.upsample_list[i](x, up_x, unq_inv)
            x = replace_feature(x, norm(x.features))
            i = i + 1
        return x

    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        embed_layer = embed_layer
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2

        if normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out