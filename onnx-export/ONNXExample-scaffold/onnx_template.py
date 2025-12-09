import argparse
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData
from typing import Dict, Optional
import torch.nn.functional as F
import os
from loadply import load_3dgs_from_ply, load_scaffold_gs_from_ply
MAX_N = 4_000_000  # 最大点数
from einops import repeat

import onnx
from onnx import TensorProto

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
            "sigmoid": nn.Sigmoid,
            "softmax": lambda: nn.Softmax(dim=1)  # Softmax 需要 dim 参数
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
    一个最小可用的"常量输出"模块。
    - 输入: camera(B,16) 和 time(B,1)，当前版本忽略
    - 输出: positions(N,3), scales(N,3), rotations(N,4), colors(N,3), opacity(N,1)
    """
    def __init__(self, anchor_data, mlp_path: str, cfg_args=None):
        super().__init__()
        # 注册为 buffer，这样导出 ONNX 时会作为 initializers 存在于模型里
        self.register_buffer("_anchor_feat", torch.from_numpy(anchor_data["anchor_feats"]).float())
        self.register_buffer("_offset", torch.from_numpy(anchor_data["offsets"]).float().contiguous())
        self.register_buffer("_anchor", torch.from_numpy(anchor_data["anchors"]).float())
        self.register_buffer("_scaling", torch.from_numpy(anchor_data["scales"]).float())
        self.register_buffer("_rotation", torch.from_numpy(anchor_data["rotations"]).float())
        self.register_buffer("_opacity", torch.from_numpy(anchor_data["opacity"]).float())
        
        # 从 cfg_args 读取参数，如果没有提供则使用默认值
        if cfg_args is not None:
            self.feat_dim = getattr(cfg_args, 'feat_dim', 32)
            self.appearance_dim = 0
            self.n_offsets = getattr(cfg_args, 'n_offsets', 10)
            self.use_feat_bank = getattr(cfg_args, 'use_feat_bank', False)
            self.add_opacity_dist = getattr(cfg_args, 'add_opacity_dist', False)
            self.add_cov_dist = getattr(cfg_args, 'add_cov_dist', False)
            self.add_color_dist = getattr(cfg_args, 'add_color_dist', False)
        else:
            # 默认值
            self.feat_dim = 32
            self.appearance_dim = 32
            self.n_offsets = 10
            self.use_feat_bank = False
            self.add_opacity_dist = False
            self.add_cov_dist = False
            self.add_color_dist = False
        
        # 根据参数计算 MLP 的输入/输出维度
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        opacity_input_dim = self.feat_dim + 3 + self.opacity_dist_dim
        opacity_output_dim = self.n_offsets
        
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        cov_input_dim = self.feat_dim + 3 + self.cov_dist_dim
        cov_output_dim = 7 * self.n_offsets
        
        self.color_dist_dim = 1 if self.add_color_dist else 0
        color_input_dim = self.feat_dim + 3 + self.color_dist_dim + self.appearance_dim
        color_output_dim = 3 * self.n_offsets
        
        # 加载 jit mlp
        sd = torch.jit.load(os.path.join(mlp_path, "opacity_mlp.pt")).state_dict()
        # 用纯 Linear 的同结构 MLP
        self.mlp_opacity = PlainMLP(opacity_input_dim, [self.feat_dim], opacity_output_dim, act="relu", out_act="tanh")
        # 直接加载（前提：层命名/维度一致；不一致就用下面"逐层拷权重"）
        self.mlp_opacity.load_state_dict(sd, strict=False)  # strict=True 要求键完全对齐
        self.mlp_opacity.eval()
        
        sd = torch.jit.load(os.path.join(mlp_path, "color_mlp.pt")).state_dict()
        # 用纯 Linear 的同结构 MLP
        self.mlp_color = PlainMLP(color_input_dim, [self.feat_dim], color_output_dim, act="relu", out_act="sigmoid")
        # 直接加载（前提：层命名/维度一致；不一致就用下面"逐层拷权重"）
        self.mlp_color.load_state_dict(sd, strict=False)  # strict=True 要求键完全对齐
        self.mlp_color.eval()
        
        sd = torch.jit.load(os.path.join(mlp_path, "cov_mlp.pt")).state_dict()
        # 用纯 Linear 的同结构 MLP
        self.mlp_cov = PlainMLP(cov_input_dim, [self.feat_dim], cov_output_dim, act="relu")
        # 直接加载（前提：层命名/维度一致；不一致就用下面"逐层拷权重"）
        self.mlp_cov.load_state_dict(sd, strict=False)  # strict=True 要求键完全对齐
        self.mlp_cov.eval()
        
        # 如果使用 feat_bank，从文件加载 mlp_feature_bank
        if self.use_feat_bank:
            feature_bank_input_dim = 3 + 1  # 3 (position) + 1 (time or other)
            feature_bank_output_dim = 3
            sd = torch.jit.load(os.path.join(mlp_path, "feature_bank_mlp.pt")).state_dict()
            # 用纯 Linear 的同结构 MLP
            self.mlp_feature_bank = PlainMLP(feature_bank_input_dim, [self.feat_dim], feature_bank_output_dim, act="relu", out_act="softmax")
            # 直接加载（前提：层命名/维度一致；不一致就用下面"逐层拷权重"）
            self.mlp_feature_bank.load_state_dict(sd, strict=False)  # strict=True 要求键完全对齐
            self.mlp_feature_bank.eval()
        
        
        # self.mlp_opacity = self._load_traced_sequential_as_module(os.path.join(mlp_path, "opacity_mlp.pt")).eval()
        # self.mlp_cov     = self._load_traced_sequential_as_module(os.path.join(mlp_path, "cov_mlp.pt")).eval()
        # self.mlp_color   = self._load_traced_sequential_as_module(os.path.join(mlp_path, "color_mlp.pt")).eval()





    def forward(self, camera: torch.Tensor, proj: torch.Tensor, time: torch.Tensor):
        """
        输入:
          camera: (B,16)，包含 extrinsics，假设最后3维是相机中心
          proj: (B,16)，暂时不用
          time: (B,1)，暂时不用
        输出:
          gauss_fixed, sh_fixed, num_points
        """
        device = self._anchor.device
        B = camera.shape[0]
        N, K = self._offset.shape[0], self._offset.shape[1]  # N anchors, K offsets




        view_cm =  camera.view(4,4)        # 列主序
        proj_cm =  proj.view(4,4)          # 列主序


        proj_cm = proj_cm.to(dtype=torch.float32)
        view_cm = view_cm.to(dtype=torch.float32)

        camera_centers = camera_center_from_view_cm(view_cm).to(device)
        camera_center = camera_centers[0:1, :]   
         # (1,3) 

        print("------camera_center")
        print(camera_center)

        mask = in_frustum_mask(self._anchor, view_cm, proj_cm,
                            near=0.01, far=1000.0, pad=1.1)  # 适当给点边界余量

        mask_f = mask.to(self._anchor.dtype).view(-1, 1)        # (N,1)
        # 你后面把 N 展成 (N*K)，这里需要 repeat 一下
        mask_fk = mask_f.repeat_interleave(K, dim=0)            # (N*K,1)




        anchors_kept = self._anchor#[mask]
        feats_kept   = self._anchor_feat#[mask]



        # === scaffold 解码 ===



   
        cc = camera_center#.expand(anchors_kept.shape[0], 3)  # (N,3)
        ob = anchors_kept - cc                           # (N,3)
        ob_view = normalize_no_div(ob, dim=1, eps=1e-12) # (N,3) 归一化方向向量
        ob_dist = torch.norm(ob, dim=1, keepdim=True)     # (N,1) 距离
        
        # print(ob_view)
        # print("---ob_view")
        feat = feats_kept  # (N,C)
        
        # 如果使用 feat_bank，进行多分辨率特征融合
        if self.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [N, 3+1]
            bank_weight = self.mlp_feature_bank(cat_view).unsqueeze(dim=1)  # [N, 1, 3]
            
            # multi-resolution feat
            feat = feat.unsqueeze(dim=-1)  # [N, C, 1]
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                   feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                   feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1)  # [N, C]
        
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)    # [N, c+3]
        
        # 根据 add_opacity_dist 决定使用哪个输入
        if self.add_opacity_dist:
            neural_opacity = self.mlp_opacity(cat_local_view)  # [N, k]
        else:
            neural_opacity = self.mlp_opacity(cat_local_view_wodist)  # [N, k]
        
        neural_opacity = neural_opacity.view(-1, 1)  # (NK,1)
        g = (neural_opacity > 0.0).to(neural_opacity.dtype)  # (NK,1) 用 float 掩码，避免 bool 流经图
        g = g * mask_fk

        # 根据 add_cov_dist 决定使用哪个输入
        if self.add_cov_dist:
            scale_rot = self.mlp_cov(cat_local_view)  # [N, 7*k]
        else:
            scale_rot = self.mlp_cov(cat_local_view_wodist)  # [N, 7*k]
        scale_rot = scale_rot.view(-1, 7)  # (NK,7)
        
        # color MLP 的输入维度已经包含了 appearance_dim，不需要根据 add_color_dist 判断
        # 但为了保持一致性，也可以添加判断
        if self.add_color_dist:
            cat_local_color = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
        else:
            cat_local_color = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
        color = self.mlp_color(cat_local_color).view(-1, 3)  # (NK,3)

        offsets = self._offset.view(-1, 3)      
        
        
        _scaling = torch.exp(self._scaling)                          # (NK,3)


        base = torch.cat([_scaling, anchors_kept], dim=-1)           # (N,9) 6(scale基) + 3(anchor)
        base = repeat(base, 'n c -> (n k) c', k=10)                        # (NK,9)
        scaling_repeat = base[:, :6]                                       # (NK,6)
        repeat_anchor  = base[:, 6:9]                                      # (NK,3)


        scaling = (scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])) \
                    .clamp_(1e-4, 1e1)

        rot = onnx_safe_normalize(scale_rot[:, 3:7], dim=-1, eps=1e-12)

        xyz     = repeat_anchor + offsets * scaling_repeat[:, :3]          # (NK,3)

        opacity  = neural_opacity * g 
        color    = color * g                                               # 可选：把无效点颜色置零
        scaling  = scaling * g                                             # 可选：把无效点尺度置零



        packed = compress_gaussians_torch(
            positions=xyz,
            scales=scaling,
            rotations=rot,
            opacity=opacity,
            colors=color,
        )

        gaussian_f16 = packed["gaussian_f16"]  # (N,10)  f16
        sh_f16       = packed["sh_f16"]        # (N,48 or 3)  f16
        sh_fixed4 = F.pad(sh_f16, (0, 1), value=0)
        num_points = torch.tensor([gaussian_f16.shape[0]], dtype=torch.int32)
        return (gaussian_f16.to(torch.float16), sh_fixed4, num_points)
        z_g = (gaussian_f16[:1, :1] - gaussian_f16[:1, :1]).reshape(())  # scalar 0, dtype跟随下行
        # z_g = z_g.to(gaussian_f16.dtype)

        z_s = (sh_f16[:1, :1] - sh_f16[:1, :1]).reshape(())
        # z_s = z_s.to(sh_f16.dtype)

        GAUSS_D = 10
        COLOR_D = 3  # 你这次走的是 RGB 分支；如果以后要走 SH48 就写 48

        # A. 继续用 expand，但 shape 全常量（避免 Shape/Slice/Gather 链）
        gauss_tail = z_g.expand(MAX_N, GAUSS_D)
        sh_tail    = z_s.expand(MAX_N, COLOR_D)

        
        
        gauss_fixed = torch.cat([gaussian_f16, gauss_tail], dim=0)[:MAX_N, :]
        sh_fixed    = torch.cat([sh_f16,       sh_tail],    dim=0)[:MAX_N, :]
        sh_fixed4 = F.pad(sh_fixed, (0, 1), value=0)  # 在最后一维右侧补 1 个元素
        print(sh_fixed4.shape)
        return (gauss_fixed.to(torch.float16), sh_fixed4, num_points)


def export_onnx(ply_path: Path, out_path: Path, opset: int = 17, cfg_args=None, mlp_path: str = "./iteration_30000"):
    data = load_scaffold_gs_from_ply(ply_path,True)
    
    model = GaussianSetModule(data, mlp_path, cfg_args=cfg_args).eval()

    dummy_camera = torch.zeros(4, 4, dtype=torch.float32)
    dummy_proj = torch.zeros(4, 4, dtype=torch.float32)
    dummy_time = torch.zeros(1, dtype=torch.float32)

    input_names = ['camera', 'proj','time']
    output_names = ['gaussian_f16', 'color_rgb', 'num_points'] 
    dynamic_axes = {
        # 'camera': {0: 'batch'},
        # 'time':   {0: 'batch'},
        # 'gaussian_f16': {0: 'num_points', 1: 'gauss_fields'},  # (N,10)
        # 'sh_f16':       {0: 'num_points', 1: 'sh_packed'},     # (N,48)
    }

    torch.onnx.export(
        model,   # 直接用原始 nn.Module
        (dummy_camera, dummy_proj, dummy_time),
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
    parser.add_argument("--ply", type=str, default="./iteration_30000/point_cloud.ply", help="3DGS PLY 文件路径")
    parser.add_argument("--out", type=str, default="gaussians3d_house.onnx", help="导出的 ONNX 路径")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本 (建议 17+)")
    parser.add_argument("--cfg_args", type=str, default=None, help="配置文件路径 (例如: output/small_city/cfg_args)")
    parser.add_argument("--mlp_path", type=str, default=None, help="MLP 模型文件路径 (默认使用 --ply 所在的文件夹)")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    out_path = Path(args.out)
    if not ply_path.exists():
        raise FileNotFoundError(f"未找到 PLY: {ply_path}")
    
    # 如果未指定 mlp_path，使用 --ply 所在的文件夹
    if args.mlp_path is None:
        mlp_path = str(ply_path.parent)
    else:
        mlp_path = args.mlp_path

    # 解析 cfg_args 文件
    cfg_args = None
    if args.cfg_args:
        cfg_args_path = Path(args.cfg_args)
        if not cfg_args_path.exists():
            raise FileNotFoundError(f"未找到配置文件: {cfg_args_path}")
        
        print(f"读取配置文件: {cfg_args_path}")
        with open(cfg_args_path, 'r') as f:
            cfgfile_string = f.read()
        # 使用 eval 解析 Namespace 对象，确保 Namespace 在全局命名空间中
        eval_globals = globals().copy()
        eval_globals['Namespace'] = Namespace
        cfg_args = eval(cfgfile_string, eval_globals)
        print(f"已加载配置参数: feat_dim={cfg_args.feat_dim}, appearance_dim={cfg_args.appearance_dim}, "
              f"n_offsets={cfg_args.n_offsets}, use_feat_bank={cfg_args.use_feat_bank}, "
              f"add_opacity_dist={cfg_args.add_opacity_dist}, add_cov_dist={cfg_args.add_cov_dist}, "
              f"add_color_dist={cfg_args.add_color_dist}")

    print(f"使用 MLP 路径: {mlp_path}")
    export_onnx(ply_path, out_path, opset=args.opset, cfg_args=cfg_args, mlp_path=mlp_path)

    import torch, onnx
    from onnx import shape_inference

    m = onnx.load(out_path)
    m = shape_inference.infer_shapes(m)

    
    onnx.save(m, out_path)

    m = onnx.load(args.out)
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
