import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement
from typing import Dict, Optional
import torch.nn.functional as F
import os
import warnings
warnings.filterwarnings("ignore")

import itertools
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

from loadply import load_3dgs_from_ply, load_scaffold_gs_from_ply, load_4dgs_from_ply
MAX_N = 4_000_000  # 最大点数
from einops import repeat

import onnx
from onnx import TensorProto

def safe_min(t):
    return torch.min(t[~torch.isnan(t)]) if torch.isnan(t).any() else torch.min(t)

def safe_max(t):
    return torch.max(t[~torch.isnan(t)]) if torch.isnan(t).any() else torch.max(t)

def safe_mean(t):
    return torch.mean(t[~torch.isnan(t)]) if torch.isnan(t).any() else torch.mean(t)

def check_tensor(name, t):
    print(f"--- {name} ---")
    print(" shape:", t.shape)
    print(" min:", safe_min(t).item())
    print(" max:", safe_max(t).item())
    print(" mean:", safe_mean(t).item())
    print(" has_nan:", torch.isnan(t).any().item())
    print(" has_inf:", torch.isinf(t).any().item())

@torch.no_grad()
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4)  [w,x,y,z]
    """
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-12)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    r00 = ww + xx - yy - zz
    r01 = 2*(xy - wz)
    r02 = 2*(xz + wy)
    r10 = 2*(xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2*(yz - wx)
    r20 = 2*(xz - wy)
    r21 = 2*(yz + wx)
    r22 = ww - xx - yy + zz
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)  # (...,3,3)
    return R


@torch.no_grad()
def build_cov_from_scales_quat(scales_lin: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    scales_lin: (N,3) 已 exp 的线性尺度
    quat_wxyz: (N,4) [w,x,y,z]
    return: (N,6) [xx, xy, xz, yy, yz, zz]
    """
    R = quat_to_rotmat(quat_wxyz)            # (N,3,3)
    s2 = (scales_lin * scales_lin)           # (N,3)
    # 列缩放: R @ diag(s2) 等价于把 R 的每一列乘以对应的 s2[j]
    R_scaled = R * s2.unsqueeze(-2)          # (N,3,3) * (N,1,3) -> (N,3,3)
    COV = R_scaled @ R.transpose(-1, -2)     # (N,3,3)

    xx = COV[:, 0, 0]; xy = COV[:, 0, 1]; xz = COV[:, 0, 2]
    yy = COV[:, 1, 1]; yz = COV[:, 1, 2]
    zz = COV[:, 2, 2]
    return torch.stack([xx, xy, xz, yy, yz, zz], dim=-1)  # (N,6)


def batch_quaternion_multiply(q1, q2):
    """
    Multiply batches of quaternions.
    
    Args:
    - q1 (torch.Tensor): A tensor of shape [N, 4] representing the first batch of quaternions.
    - q2 (torch.Tensor): A tensor of shape [N, 4] representing the second batch of quaternions.
    
    Returns:
    - torch.Tensor: The resulting batch of quaternions after applying the rotation.
    """
    # Calculate the product of each quaternion in the batch
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Combine into new quaternions
    q3 = torch.stack((w, x, y, z), dim=1)
    
    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=1, keepdim=True)
    
    return norm_q3

@torch.no_grad()
def pack_gaussian_f16(
    positions: torch.Tensor,     # (N,3) f32
    scales_log: torch.Tensor,    # (N,3) f32, 需要 exp
    rotations: torch.Tensor,     # (N,4) f32, [w,x,y,z]
    opacity: torch.Tensor        # (N,1) or (N,) f32, 需要 sigmoid
) -> torch.Tensor:
    """
    输出 (N,10) f16: [px,py,pz, sigmoid(op), m00,m01,m02,m11,m12,m22]
    """
    N = positions.shape[0]
    pos = positions.float()
    # op  = torch.sigmoid(opacity.view(-1, 1).float())   # (N,1)
    op  = (opacity.view(-1, 1).float())   # (N,1)
    # scales_lin = torch.exp(scales_log.float())             # (N,3) 与前端一致
    scales_lin = scales_log.float() 
    cov6 = build_cov_from_scales_quat(scales_lin, rotations.float())  # (N,6)

    gauss_f32 = torch.cat([pos, op, cov6], dim=1)         # (N,10) f32
    return gauss_f32# .to(torch.float16)                    # (N,10) f16


import torch

def construct_list_of_attributes(feature_dc_shape, feature_rest_shape, scaling_shape,rotation_shape):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(feature_dc_shape[1]*feature_dc_shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(feature_rest_shape[1]*feature_rest_shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling_shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation_shape[1]):
        l.append('rot_{}'.format(i))
    # breakpoint()
    return l
def init_3DGaussians_ply(points, scales, rotations, opactiy, shs, feature_shape):
    xyz = points.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    feature_dc = shs[:,0:feature_shape[0],:]
    feature_rest = shs[:,feature_shape[0]:,:]
    f_dc = shs[:,:feature_shape[0],:].detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
    # breakpoint()
    f_rest = shs[:,feature_shape[0]:,:].detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opactiy.detach().cpu().numpy()
    scale = scales.detach().cpu().numpy()
    rotation = rotations.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(feature_dc.shape, feature_rest.shape, scales.shape, rotations.shape)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    # breakpoint()
    return PlyData([el])

@torch.no_grad()
def pack_sh_to_48_f16(colors: torch.Tensor) -> torch.Tensor:
    """
    输入 colors: (N, K) f32/任意，按约定顺序：
      [R_dc, G_dc, B_dc, R_rest(<=15), G_rest(<=15), B_rest(<=15), ...]
    输出: (N,48) f16，布局为交错：
      [R_dc, G_dc, B_dc, (R1,G1,B1), (R2,G2,B2), ..., (R15,G15,B15)]
    K 可以小于 3（自动补 0），也可以 >48（会按规则裁剪）
    """
    device = colors.device
    colors = colors.float()
    N, K = colors.shape if colors.ndim == 2 else (colors.shape[0], 0)

    out = torch.zeros((N, 48), dtype=torch.float16, device=device)

    # DC 3 项（若 K<3，切片为空，赋值安全）
    if K > 0:
        out[:, 0:3] = colors[:, 0:3].to(torch.float16)

    # 每通道最多 15 个高阶系数（共 45）
    if K > 3:
        nr_full = (K - 3) // 3
        nr = min(15, int(nr_full))
        if nr > 0:
            r_rest = colors[:, 3:3+nr]                 # (N, nr)
            g_rest = colors[:, 3+nr:3+2*nr]            # (N, nr)
            b_rest = colors[:, 3+2*nr:3+3*nr]          # (N, nr)
            interleaved = torch.stack([r_rest, g_rest, b_rest], dim=2) \
                              .reshape(N, nr*3)        # (N, 3*nr)
            out[:, 3:3+nr*3] = interleaved.to(torch.float16)

    return out


@torch.no_grad()
def prepare_color_buffers(colors: torch.Tensor):
    """
    统一入口：根据 K 自动选择 RGB 或 SH48。
    返回:
      sh48_f16: (N,48) or None
      rgb_f16:  (N,3)  or None
      color_mode: 'rgb' | 'sh'
      color_channels: 3 | 48
    说明：
      - K==3/4 → 走 RGB，丢弃 alpha。
      - 其他 → 走 SH48（使用 pack_sh_to_48_f16）
    """
    colors = colors.to(torch.float32, copy=False)
    N, K = colors.shape

    # if K <= 0:
    #     # 空输入：给出全零 SH48
    #     sh48 = torch.zeros((N, 48), dtype=torch.float16, device=colors.device)
    #     return sh48, None, 'sh', 48

    if K == 3 or K == 4:
        rgb = colors[:, :3].to(torch.float16).contiguous()
        return rgb

    # 其他情况按“分通道再交错”的约定打成 48
    sh48 = pack_sh_to_48_f16(colors)
    return sh48





@torch.no_grad()
def compress_gaussians_torch(
    *,
    positions: torch.Tensor,                  # (N,3) f32
    scales: Optional[torch.Tensor] = None,    # (N,3) f32, 对应 log-scale
    rotations: Optional[torch.Tensor] = None, # (N,4) f32 [w,x,y,z]
    opacity: Optional[torch.Tensor] = None,   # (N,1) or (N,) f32
    colors: Optional[torch.Tensor] = None     # (N,K) f32
) -> Dict[str, torch.Tensor]:
    """
    直接打包成渲染端所需：
      - gaussian_f16: (N,10) f16  [px,py,pz,sigmoid(op), m00,m01,m02,m11,m12,m22]
      - sh_f16:       (N,48) f16  [R_dc,G_dc,B_dc, R1,G1,B1, ...]（固定 48 个 f16）
    """

    print('---')
    print(colors.shape)
    gaussian_f16 = pack_gaussian_f16(positions, scales, rotations, opacity)
    sh_f16 = prepare_color_buffers(colors)
    print('---')
    print(sh_f16.shape)
    return {"gaussian_f16": gaussian_f16, "sh_f16": sh_f16}


# class TorchScriptWrapper(nn.Module):
#     def __init__(self, script_module):
#         super().__init__()
#         self.script_module = script_module  # 注册为子模块

#     def forward(self, x):
#         return self.script_module(x)

class PlainMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act="relu", out_act=None):
        """
        in_dim: 输入维度
        hidden: 隐藏层列表，例如 [32, 64]
        out_dim: 输出维度
        act: 隐藏层激活函数 ("relu" / "tanh" / "sigmoid")
        out_act: 输出层激活函数 (同上, 默认 None 表示不加)
        """
        super().__init__()

        # 激活函数映射表
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }

        if act not in act_map:
            raise ValueError(f"Unsupported act={act}, must be one of {list(act_map.keys())}")

        layers = []
        dim = in_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), act_map[act]()]
            dim = h

        layers += [nn.Linear(dim, out_dim)]

        if out_act is not None:
            if out_act not in act_map:
                raise ValueError(f"Unsupported out_act={out_act}, must be one of {list(act_map.keys())}")
            layers += [act_map[out_act]()]

        self._modules.update({str(i): layer for i, layer in enumerate(layers)})

    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
    
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size

        self.grid = nn.Parameter(torch.ones([1, channels, *world_size]))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        # if self.channels == 1:
            # out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.ones([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))
    def set_aabb(self, xyz_max, xyz_min):
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size}'

def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid

        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)   
 

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            # breakpoint()
            shs = shs_emb*mask.unsqueeze(-1) + dshs

        return pts, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1

        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()



@torch.no_grad()
def onnx_safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    # 用基础算子拼出：sqrt(clamp(sum(x^2), eps^2))，再 x/denom
    # 导出到 ONNX 会是：Mul → ReduceSum → Clip → Sqrt → Div，保留 eps 语义
    n2    = (x * x).sum(dim=dim, keepdim=True)
    denom = torch.sqrt(torch.clamp(n2, min=eps * eps))
    return x / denom



def in_frustum_mask(points: torch.Tensor,
                    view_cm: torch.Tensor,
                    proj_cm: torch.Tensor,
                    near: float = 0.01,
                    far:  float = 100.0,
                    pad:  float = 1.0):
    """
    points:   (#points, 3) 世界坐标
    view_cm:  (4,4)  列主序(=转置)的 world-to-camera 矩阵  [[R^T, 0],[t^T, 1]]
    proj_cm:  (4,4)  列主序(=转置)的投影矩阵
    near/far: 以“相机坐标系 z”为准的前后裁剪距离
    pad:      >1 表示给左右上下各放大一点（例如 1.05~1.3）
    返回:     (#points,) 的 bool mask
    """
    device = points.device
    dtype  = points.dtype

    # 转回“行主序”便于计算
    V = view_cm.t().to(dtype=dtype, device=device)  # (4,4) world->camera
    P = proj_cm.t().to(dtype=dtype, device=device)  # (4,4) camera->clip

    N = points.shape[0]
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    Xw = torch.cat([points, ones], dim=1)           # (N,4)

    # 相机空间（view space）
    Xv = Xw @ V.t()                                 # (N,4)
    z_view = Xv[:, 2]                               # (N,)

    # 裁剪 z：只保留 near<z<far
    z_ok = (z_view > near) & (z_view < far)

    # 投影到 clip 空间
    Xc = Xv @ P.t()                                 # (N,4)
    x_c, y_c, w_c = Xc[:, 0], Xc[:, 1], Xc[:, 3]

    # 使用 clip 不等式：|x| <= pad * w, |y| <= pad * w
    # （这样不依赖于 OpenGL(-1..1) 或 D3D(0..1) 的 NDC z 规范）
    w_pos = w_c.abs().clamp_min(1e-7)

    # —— 全部用 float(0/1)，避免 And/Or —— #
    cx = (x_c.abs() <= pad * w_pos).to(dtype)          # (N,)
    cy = (y_c.abs() <= pad * w_pos).to(dtype)          # (N,)
    cz1 = (z_view >  near).to(dtype)                   # (N,)
    cz2 = (z_view <  far ).to(dtype)                   # (N,)

    mask_f = cx * cy * cz1 * cz2                       # (N,)
    # 转 bool 用比较（Greater），WebGPU 支持
    return (mask_f > 0.5)




def camera_center_from_view_cm(camera_flat: torch.Tensor) -> torch.Tensor:
    M   = camera_flat.view(-1, 4, 4)     # (B,4,4)
    R_T = M[:, :3, :3]
    t   = M[:, 3, :3].unsqueeze(-1)
    C   = -(R_T @ t).squeeze(-1)         # (B,3)
    return C

def normalize_no_div(x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    # inv_len = (sum(x^2)+eps^2)^(-1/2) 用 Pow 实现，避免 Div
    n2 = (x * x).sum(dim=dim, keepdim=True).clamp_min(eps * eps)
    inv = torch.pow(n2, -0.5)
    return x * inv


class GaussianSetModule(nn.Module):
    """
    一个最小可用的“常量输出”模块。
    - 输入: camera(B,16) 和 time(B,1)，当前版本忽略
    - 输出: positions(N,3), scales(N,3), rotations(N,4), colors(N,3), opacity(N,1)
    """
    def __init__(self, ply_data, mlp_path: str):
        super().__init__()
        # 注册为 buffer，这样导出 ONNX 时会作为 initializers 存在于模型里

        for k, np_arr in ply_data.items():
            t = torch.from_numpy(np_arr.astype(np.float32))
            self.register_buffer(k, t, persistent=True)
            self.register_buffer('y_axis', torch.tensor([0.0, 1.0, 0.0]).view(1, 3))

        # 加载 4dgs mlp
        import argparse
        args_path = os.path.join(mlp_path, '../../cfg_args')

        with open(args_path, "r") as f:
            s = f.read().strip()

        args = eval(s, {"__builtins__": {}}, {"Namespace": argparse.Namespace})

        self._deformation = deform_network(args)

        aabb = args.grid_aabb
        # self._deformation.deformation_net.set_aabb([1.29982098, 1.29990645, 1.29988719],[-1.29980838, -1.29981163, -1.29872349])
        self._deformation.deformation_net.set_aabb(aabb[0], aabb[1])
        
        print("loading model from exists{}".format(mlp_path))
        weight_dict = torch.load(os.path.join(mlp_path, "deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.positions.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.positions.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(mlp_path, "deformation_table.pth")):
            print('deformation_table')
            self._deformation_table = torch.load(os.path.join(mlp_path, "deformation_table.pth"), map_location="cuda")
        if os.path.exists(os.path.join(mlp_path, "deformation_accum.pth")):
            print('deformation_accum')
            self._deformation_accum = torch.load(os.path.join(mlp_path, "deformation_accum.pth"), map_location="cuda")
        self.max_radii2D = torch.zeros((self.positions.shape[0]), device="cuda")


    def forward(self, time: torch.Tensor):
        """
        输入:
          time: (B,1)
        输出:
          gauss_fixed, sh_fixed, num_points
        """
        device = self.positions.device
        time = time.to(self.positions.device).repeat(self.positions.shape[0],1)
        print(self.features_dc.shape, self.features_extra.shape)
        shs = torch.cat((self.features_dc.transpose(1, 2), self.features_extra.transpose(1, 2)), dim=1)

        xyz, scaling, rot, opacity, shs = \
            self._deformation(self.positions, self.scales, self.rotations, self.opacity, shs, time)

        # gs_ply = init_3DGaussians_ply(xyz, scaling, rot, opacity, shs, [1, 15])
        # gs_ply.write(os.path.join("time.ply"))

        # scaling_check = torch.exp(scaling)
        # shs_check = shs.reshape(xyz.shape[0], -1)
        
        # 条件 1：scaling 的任意一维不能 > 0.3
        # 等价于：这一行所有维度都 <= 0.3
        # mask_scaling = (scaling_check <= 1.).all(dim=1)

        # 条件 2：shs 的任意一维不能 |x| > 3
        # 等价于：这一行所有维度都 |x| <= 3
        # mask_shs = (shs_check.abs() <= 3).all(dim=1)

        # 综合条件：两个都满足才保留
        # mask = mask_scaling & mask_shs

        # print("保留点数:", mask.sum().item(), "/", mask.numel())
        # print("删除点数:", (~mask).sum().item())

        # 用同一个 mask 过滤所有属性
        # xyz     = xyz[mask]
        # scaling = scaling[mask]
        # rot     = rot[mask]
        # opacity = opacity[mask]
        # shs     = shs[mask]

        # gs_ply = init_3DGaussians_ply(xyz, scaling, rot, opacity, shs, [1, 15])
        # gs_ply.write(os.path.join("time.ply"))
        # assert False

        scaling = torch.exp(scaling)
        rot = torch.nn.functional.normalize(rot)
        opacity = torch.sigmoid(opacity)
        shs = shs.reshape(xyz.shape[0], -1)

        # check_tensor("xyz0", self.positions)
        # check_tensor("scaling0", self.scales)
        # check_tensor("rot0", self.rotations)
        # check_tensor("opacity0", self.opacity)
        # check_tensor("shs0", shs)

        # check_tensor("xyz", xyz)
        # check_tensor("scaling", scaling)
        # check_tensor("rot", rot)
        # check_tensor("opacity", opacity)
        # check_tensor("shs", shs)
        # assert False

        packed = compress_gaussians_torch(
            positions=xyz,
            scales=scaling,
            rotations=rot,
            opacity=opacity,
            colors=shs,
        )

        gaussian_f16 = packed["gaussian_f16"]  # (N,10)  f16
        sh_f16       = packed["sh_f16"]        # (N,48 or 3)  f16

        z_g = (gaussian_f16[:1, :1] - gaussian_f16[:1, :1]).reshape(())  # scalar 0, dtype跟随下行
        # z_g = z_g.to(gaussian_f16.dtype)

        z_s = (sh_f16[:1, :1] - sh_f16[:1, :1]).reshape(())
        # z_s = z_s.to(sh_f16.dtype)

        GAUSS_D = 10
        COLOR_D = 48  # 你这次走的是 RGB 分支；如果以后要走 SH48 就写 48

        # A. 继续用 expand，但 shape 全常量（避免 Shape/Slice/Gather 链）
        gauss_tail = z_g.expand(MAX_N, GAUSS_D)
        sh_tail    = z_s.expand(MAX_N, COLOR_D)

        
        num_points = torch.tensor([gaussian_f16.shape[0]], dtype=torch.int32)
        gauss_fixed = torch.cat([gaussian_f16, gauss_tail], dim=0)[:MAX_N, :]
        sh_fixed    = torch.cat([sh_f16,       sh_tail],    dim=0)[:MAX_N, :]
        # sh_fixed4 = F.pad(sh_fixed, (0, 1), value=0)  # 在最后一维右侧补 1 个元素
        return (gauss_fixed.to(torch.float16), sh_fixed, num_points)


def export_onnx(ply_path: Path, out_path: Path, opset: int = 17):

    data = load_4dgs_from_ply(ply_path)
    
    model = GaussianSetModule(data, os.path.dirname(ply_path)).to("cuda").eval()

    dummy_camera = torch.zeros(4, 4, dtype=torch.float32)
    dummy_proj = torch.zeros(4, 4, dtype=torch.float32)
    dummy_time = torch.ones(1, dtype=torch.float32)

    input_names = ['time']
    output_names = ['gaussian_f16', 'color_sh', 'num_points']
    dynamic_axes = {
        # 'camera': {0: 'batch'},
        # 'time':   {0: 'batch'},
        # 'gaussian_f16': {0: 'num_points', 1: 'gauss_fields'},  # (N,10)
        # 'sh_f16':       {0: 'num_points', 1: 'sh_packed'},     # (N,48)
    }

    torch.onnx.export(
        model,   # 直接用原始 nn.Module
        (dummy_time),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=False,
        export_params=True,
    )

    print(f"✅ Exported ONNX to: {out_path.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default="./ckpt/iteration_20000/point_cloud.ply", help="3DGS PLY 文件路径")
    parser.add_argument("--out", type=str, default="gaussians4d.onnx", help="导出的 ONNX 路径")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本 (建议 17+)")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    out_path = Path(args.out)
    if not ply_path.exists():
        raise FileNotFoundError(f"未找到 PLY: {ply_path}")

    export_onnx(ply_path, out_path, opset=args.opset)



    m = onnx.load("gaussians4d.onnx")
    for init in m.graph.initializer:
        assert init.data_type != TensorProto.FLOAT16, f"initializer {init.name} is fp16"
    for node in m.graph.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name=="to" and a.i==TensorProto.FLOAT16:
                    print(node.name)
                    #raise RuntimeError(f"Cast to fp16 at node {node.name}")
    # print("✅ graph is pure fp32")



if __name__ == "__main__":
    main()
