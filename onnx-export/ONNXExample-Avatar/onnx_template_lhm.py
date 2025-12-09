# # --- 放到 G:\ONNXExample\onnx_template.py 顶部第一行 ---
# import os
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'   # 4090=8.9
# # ---------------------------------------------------------



import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os
import json
import torch.nn as nn
from plyfile import PlyData
from typing import Dict, Optional
import torch.nn.functional as F
# from utils.loadply import load_3dgs_from_ply, load_gauhuman_ply
# from knn_cuda import KNN
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from natsort import natsorted

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix_refine
# from utils.mlp_delta_weight_lbs import LBSOffsetDecoder
from utils.smpl_x_voxel_dense_sampling import SMPLXVoxelMeshModel, batch_rigid_transform

import onnx
from onnx import shape_inference, checker

def rename_value_everywhere(model, old, new):
    # graph io/value_info
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name == old:
            vi.name = new

    # initializers
    for init in model.graph.initializer:
        if init.name == old:
            init.name = new

    # all node inputs/outputs
    for node in model.graph.node:
        node.input[:]  = [new if x == old else x for x in node.input]
        node.output[:] = [new if x == old else x for x in node.output]

def fix_output_name(path, out_index=1, desired_name="color_rgb"):
    m = onnx.load(path)

    old = m.graph.output[out_index].name
    if old != desired_name:
        rename_value_everywhere(m, old, desired_name)

    # 再做一次 shape 推断 + 合法性检查
    m = shape_inference.infer_shapes(m)
    checker.check_model(m)

    onnx.save(m, path)
    print(f"✅ renamed output[{out_index}] {old} -> {desired_name}")


MAX_N = 500_000  # 最大点数


def safe_cat(chunks, dim=1, fanin=4):
    # 把一次大 cat 拆成每层最多 fanin 个输入的多层 cat
    if len(chunks) <= fanin:
        return torch.cat(chunks, dim=dim)
    level = chunks
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), fanin):
            next_level.append(torch.cat(level[i:i+fanin], dim=dim))
        level = next_level
    return level[0]




def blend_shapes(betas, shape_disps):
    """Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum("bl,mkl->bmk", [betas, shape_disps])
    return blend_shape




def load_LHM_pth(path, max_sh_degree=3):
    

    dxdydz, xyz, rgb, opacity, scaling, rotation, transform_mat_neutral_pose, esti_shape, body_ratio, have_face = torch.load(path)

    # print(xyz.shape)
    # print(rgb.shape)
    # print(opacity.shape)
    # print(scaling.shape)
    # print(rotation.shape)

    return {
        'dxdydz': dxdydz.cpu().numpy(), 
        'positions': xyz.cpu().numpy(), 
        'colors': rgb.cpu().numpy(), 
        'opacities': opacity.cpu().numpy(), 
        'scales': scaling.cpu().numpy(), 
        'rotations': rotation.cpu().numpy(), 
        'transform_mat_neutral_pose': transform_mat_neutral_pose.cpu().numpy(), 
        'esti_shape': esti_shape.cpu().numpy()
    }




def load_camera(pose):
    intrinsic = torch.eye(3)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()

    image_width, image_height = pose["img_size_wh"]

    c2w = torch.eye(4)
    c2w = c2w.float()

    return c2w, intrinsic, image_height, image_width



def get_camera_smplx_data(smplx_path):
    with open(smplx_path) as f:
        smplx_raw_data = json.load(f)
    
    smplx_param = {
        k: torch.FloatTensor(v)
        for k, v in smplx_raw_data.items()
        if "pad_ratio" not in k
    }
    
    c2w, K, image_height, image_width = load_camera(smplx_param)
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    
    focalX = K[0, 0]
    focalY = K[1, 1]
    FovX = focal2fov(focalX, image_width)
    FovY = focal2fov(focalY, image_height)

    zfar = 1000
    znear = 0.001
    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0

    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
    projection_matrix = getProjectionMatrix_refine(torch.Tensor(K), image_height, image_width, znear, zfar).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    # smplx_param['betas'] = esti_shape
    smplx_param['expr'] = torch.zeros((100))

    return {
        'smplx_betas': smplx_param['betas'], 
        'smplx_root_pose': smplx_param['root_pose'], 
        'smplx_body_pose': smplx_param['body_pose'], 
        'smplx_jaw_pose': smplx_param['jaw_pose'], 
        'smplx_leye_pose': smplx_param['leye_pose'], 
        'smplx_reye_pose': smplx_param['reye_pose'], 
        'smplx_lhand_pose': smplx_param['lhand_pose'], 
        'smplx_rhand_pose': smplx_param['rhand_pose'], 
        'smplx_trans': smplx_param['trans'], 
        'smplx_expr': smplx_param['expr'], 
        # 'transform_mat_neutral_pose': transform_mat_neutral_pose, 
        # 'smplx_param': smplx_param, 
        # 'w2c': w2c, 
        # 'R': R, 
        # 'T': T, 
        # 'K': K, 
        'cam_FoVx': torch.tensor(FovX), 
        'cam_FoVy': torch.tensor(FovY), 
        'cam_zfar': torch.tensor(zfar), 
        'cam_znear': torch.tensor(znear), 
        'cam_trans': torch.tensor(trans), 
        'cam_scale': torch.tensor(scale), 
        'cam_image_height': torch.tensor(image_height), 
        'cam_image_width': torch.tensor(image_width), 
        'cam_world_view_transform': world_view_transform, 
        'cam_projection_matrix': projection_matrix, 
        'cam_full_proj_transform': full_proj_transform, 
        'cam_camera_center': camera_center, 
    }



def rodrigues_rotation_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """
    将旋转向量 (axis-angle) 转换为 3x3 旋转矩阵
    rotvec: [3] tensor
    return: [3,3] tensor
    """
    theta = torch.norm(rotvec)  # 旋转角度
    if theta < 1e-8:
        return torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)

    k = rotvec / theta  # 单位旋转轴 [3]
    K = torch.tensor([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ], device=rotvec.device, dtype=rotvec.dtype)

    R = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype) \
        + torch.sin(theta) * K \
        + (1 - torch.cos(theta)) * (K @ K)
    return R





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
    # op  = torch.sigmoid(opacity.reshape(-1, 1).float())   # (N,1)
    # scales_lin = torch.exp(scales_log.float())             # (N,3) 与前端一致

    op  = (opacity.reshape(-1, 1).float())   # (N,1)
    scales_lin = (scales_log.float())             # (N,3) 与前端一致
    cov6 = build_cov_from_scales_quat(scales_lin, rotations.float())  # (N,6)

    gauss_f32 = torch.cat([pos, op, cov6], dim=1)         # (N,10) f32
    return gauss_f32.to(torch.float16)                    # (N,10) f16




@torch.no_grad()
def pack_gauhuman_gaussian_f16(
    positions: torch.Tensor,     # (N,3) f32
    cov3D_precomp: torch.Tensor, # (N,6) f32
    opacity: torch.Tensor        # (N,1) or (N,) f32, 需要 sigmoid
) -> torch.Tensor:
    """
    输出 (N,10) f16: [px,py,pz, sigmoid(op), m00,m01,m02,m11,m12,m22]
    """
    N = positions.shape[0]
    pos = positions.float()
    # op  = torch.sigmoid(opacity.reshape(-1, 1).float())   # (N,1)
    op  = opacity.reshape(-1, 1).float()   # (N,1)
    # scales_lin = torch.exp(scales_log.float())             # (N,3) 与前端一致
    cov6 = cov3D_precomp  # (N,6)

    gauss_f32 = torch.cat([pos, op, cov6], dim=1)         # (N,10) f32
    return gauss_f32.to(torch.float16)                    # (N,10) f16



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
        rgb = colors.to(torch.float16).contiguous()
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
    print(positions.shape)
    print(scales.shape)
    print(rotations.shape)
    print(opacity.shape)
    print('---')

    gaussian_f16 = pack_gaussian_f16(positions, scales, rotations, opacity)
    sh_f16 = prepare_color_buffers(colors)
    print('---')
    print(colors.shape)
    print(sh_f16.shape)
    print('---')

    return {"gaussian_f16": gaussian_f16, "sh_f16": sh_f16}


def _pick_one_hot(table: torch.Tensor, idx_scalar: torch.Tensor):
    # table: (T, ...), idx_scalar: () float32，范围 [0, T-1]
    T = table.shape[0]
    ar = torch.arange(T, dtype=table.dtype, device=table.device)   # 导出为常量 initializer
    # 先把 t->idx：floor/clip 保证范围；保持 float 以避免 int64 索引路径
    i = torch.clamp(torch.floor(idx_scalar), 0, T - 1)
    # one-hot mask: (T,)
    mask = (ar == i).to(table.dtype)
    # 形状对齐后加权求和相当于选择那一行/那一帧
    expand_dims = (T,) + (1,) * (table.ndim - 1)
    return (table * mask.view(*expand_dims)).sum(dim=0)

@torch.no_grad()
def compress_gauhuman_gaussians_torch(
    *,
    positions: torch.Tensor,                  # (N,3) f32
    # scales: Optional[torch.Tensor] = None,    # (N,3) f32, 对应 log-scale
    # rotations: Optional[torch.Tensor] = None, # (N,4) f32 [w,x,y,z]
    cov3D_precomp: Optional[torch.Tensor] = None, # (N,6) f32
    opacity: Optional[torch.Tensor] = None,   # (N,1) or (N,) f32
    shs: Optional[torch.Tensor] = None     # (N,K) f32
) -> Dict[str, torch.Tensor]:
    """
    直接打包成渲染端所需：
      - gaussian_f16: (N,10) f16  [px,py,pz,sigmoid(op), m00,m01,m02,m11,m12,m22]
      - sh_f16:       (N,48) f16  [R_dc,G_dc,B_dc, R1,G1,B1, ...]（固定 48 个 f16）
    """

    print('---')
    print(positions.shape)
    print(cov3D_precomp.shape)
    print(opacity.shape)
    print(shs.shape)
    print('---')

    gaussian_f16 = pack_gauhuman_gaussian_f16(positions, cov3D_precomp, opacity)
    sh_f16 = prepare_color_buffers(shs)
    print('---')
    print(sh_f16.shape)
    print('---')

    return {"gaussian_f16": gaussian_f16, "sh_f16": sh_f16}


def pick_with_gather(table: torch.Tensor, idx_scalar: torch.Tensor):
    """
    table: (T, ...)
    idx_scalar: () float32, [0, T-1]
    返回: table[idx]
    """
    T = table.shape[0]

    # float -> 合法整数下标，仍然是张量路径
    idx = torch.floor(idx_scalar)
    idx = torch.clamp(idx, 0, T - 1)

    # 关键：强制成 int32，避免 int64 索引路径
    idx = idx.to(torch.int32)   # 或 torch.int64，看你后端支持哪个

    # index_select 的 index 必须是一维
    idx_1d = idx.view(1)        # shape: (1,)

    # 这一步在 ONNX 里一般导出成 Gather
    picked = torch.index_select(table, dim=0, index=idx_1d)  # (1, ...)

    return picked[0]   # 去掉前面的 batch 维


def get_zero_pose_human(
    SMPLX_MODEL, shape_param, device='cpu', face_offset=None, joint_offset=None, return_mesh=False
):
    smpl_x = SMPLX_MODEL.smpl_x
    batch_size = shape_param.shape[0]

    zero_pose = torch.zeros((batch_size, 3)).float().to(device)
    zero_body_pose = (
        torch.zeros((batch_size, (len(smpl_x.joint_part["body"]) - 1) * 3))
        .float()
        .to(device)
    )
    zero_hand_pose = (
        torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
        .float()
        .to(device)
    )
    zero_expr = torch.zeros((batch_size, smpl_x.expr_param_dim)).float().to(device)

    face_offset = face_offset
    joint_offset = (
        smpl_x.get_joint_offset(joint_offset) if joint_offset is not None else None
    )
    # print(SMPLX_MODEL.smplx_layer.device)
    # print(zero_pose.device)
    output = SMPLX_MODEL.smplx_layer(
        global_orient=zero_pose,
        body_pose=zero_body_pose,
        left_hand_pose=zero_hand_pose,
        right_hand_pose=zero_hand_pose,
        jaw_pose=zero_pose,
        leye_pose=zero_pose,
        reye_pose=zero_pose,
        expression=zero_expr,
        betas=shape_param,
        face_offset=face_offset,
        joint_offset=joint_offset,
    )
    joint_zero_pose = output.joints[:, : smpl_x.joint_num, :]  # zero pose human

    if not return_mesh:
        return joint_zero_pose
    else:
        raise NotImplementedError
    


class GaussianSetModule(nn.Module):
    """
    一个最小可用的“常量输出”模块。
    - 输入: camera(B,16) 和 time(B,1)，当前版本忽略
    - 输出: positions(N,3), scales(N,3), rotations(N,4), colors(N,3), opacity(N,1)
    """
    def __init__(self, ply_data: dict, motion_json):
        super().__init__()
        # 注册为 buffer，这样导出 ONNX 时会作为 initializers 存在于模型里
        for k, np_arr in ply_data.items():
            print(k, np_arr.shape)
            t = torch.from_numpy(np_arr.astype(np.float32))
            self.register_buffer(k, t, persistent=True)
            self.register_buffer('y_axis', torch.tensor([0.0, 1.0, 0.0]).view(1, 3))


        # Pre-load SMPL parameters from json files
        camera_smpl_datas = []
        for json_path in natsorted(os.listdir(motion_json)):
            cam_smpl_data = get_camera_smplx_data(os.path.join(motion_json, json_path))
            camera_smpl_datas.append(cam_smpl_data)
        stacked_camera_smpl_dict = {}
        print('---loading smplx parames---')
        for key in camera_smpl_datas[0].keys():  # 遍历所有的 key
            tensors = [d[key] for d in camera_smpl_datas]  # 收集每个字典中该 key 的 tensor
            stacked_camera_smpl_dict[key] = torch.stack(tensors, dim=0)  # 堆叠
            print(key, stacked_camera_smpl_dict[key].shape)
        for key, value in stacked_camera_smpl_dict.items():
            self.register_buffer(key, value.cuda())
        print('---completed loading smplx parames---')


        SMPLX_MODEL = SMPLXVoxelMeshModel(
            './pretrained_models/human_model_files',
            gender="neutral",
            subdivide_num=1,
            shape_param_dim=10,
            expr_param_dim=100,
            cano_pose_type=1,
            dense_sample_points=40000,
            apply_pose_blendshape=False,
        ).cuda()

        # print(SMPLX_MODEL.expr_dirs.shape)

        # assert False


        self.register_buffer("SMPLX_MODEL_is_constrain_body", SMPLX_MODEL.is_constrain_body)
        self.register_buffer("SMPLX_MODEL_expr_dirs", SMPLX_MODEL.expr_dirs)
        self.register_buffer("SMPLX_MODEL_is_rhand", SMPLX_MODEL.is_rhand)
        self.register_buffer("SMPLX_MODEL_is_lhand", SMPLX_MODEL.is_lhand)
        self.register_buffer("SMPLX_MODEL_is_face",  SMPLX_MODEL.is_face)
        self.register_buffer("SMPLX_MODEL_voxel_bbox", SMPLX_MODEL.voxel_bbox)
        self.register_buffer("SMPLX_MODEL_voxel_ws",   SMPLX_MODEL.voxel_ws)
        self.register_buffer("SMPLX_MODEL_skinning_weight", SMPLX_MODEL.skinning_weight)
        self.register_buffer("SMPLX_MODEL_shape_dirs", SMPLX_MODEL.shape_dirs)
        self.register_buffer("SMPLX_MODEL_smplx_layer_parents",
                             SMPLX_MODEL.smplx_layer.parents)
        # 计算并注册 joint_null_pose
        joint_null_pose = get_zero_pose_human(
            SMPLX_MODEL, self.smplx_betas[:2], 'cuda'
        )
        self.register_buffer("joint_null_pose", joint_null_pose.cuda())
        t_orig = torch.zeros((1, 1)).float().cuda()
        gaussian_canon_dxdydz = self.dxdydz.to(t_orig.device)
        query_points = self.positions.to(t_orig.device)
        self.colors = self.colors.to(t_orig.device).to(torch.float16)
        self.colors    = F.pad(self.colors, (0, 1), value=0).contiguous()  # 在最后一维右侧补 1 个元素

        self.register_buffer("colors_4",  self.colors)
        gaussian_canon_opacity = self.opacities.to(t_orig.device)
        gaussian_canon_scaling = self.scales.to(t_orig.device)
        gaussian_canon_rotation = self.rotations.to(t_orig.device)
        transform_mat_neutral_pose = self.transform_mat_neutral_pose.to(t_orig.device)

    def _pick_by_t_1d(self, table: torch.Tensor, t: torch.Tensor, keepdim: bool) -> torch.Tensor:
        """
        table: [T, ...]
        t: scalar in [0,1]
        返回 table[idx]，用 ONNX 友好的写法（int64 + index_select）
        keepdim=True 保留前面的 batch 维度（[1,...]）
        """
        T = table.shape[0]
        # 用 floor/clip，避免需要 Round（Web 上 Round 有时没有 GPU kernel）
        idx = (t * (T - 1)).floor()                         # float
        idx = torch.clamp(idx, 0, T - 1).to(torch.long)     # int64
        idx1 = idx.view(1)                                  # [1]
        out = torch.index_select(table, 0, idx1)            # [1, ...]
        if not keepdim:
            out = out.squeeze(0)                            # 常量 axis=0，ONNX 好处理
        return out

    def get_transform_mat_joint(
        self, transform_mat_neutral_pose, joint_zero_pose, smplx_param
    ):
        """_summary_
        Args:
            transform_mat_neutral_pose (_type_): [B, 55, 4, 4]
            joint_zero_pose (_type_): [B, 55, 3]
            smplx_param (_type_): dict
        Returns:
            _type_: _description_
        """

        # 1. 大 pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose
        root_pose = smplx_param["root_pose"]
        body_pose = smplx_param["body_pose"]
        jaw_pose = smplx_param["jaw_pose"]
        leye_pose = smplx_param["leye_pose"]
        reye_pose = smplx_param["reye_pose"]
        lhand_pose = smplx_param["lhand_pose"]
        rhand_pose = smplx_param["rhand_pose"]
        # trans = smplx_param['trans']

        # forward kinematics

        pose = safe_cat([
            root_pose.unsqueeze(1),
            body_pose,
            jaw_pose.unsqueeze(1),
            leye_pose.unsqueeze(1),
            reye_pose.unsqueeze(1),
            lhand_pose,
            rhand_pose,
        ], dim=1, fanin=3)  # [B, 55, 3]
        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]
        posed_joints, transform_mat_joint_2 = batch_rigid_transform(
            pose[:, :, :, :], joint_zero_pose[:, :, :], self.SMPLX_MODEL_smplx_layer_parents
        )
        transform_mat_joint_2 = transform_mat_joint_2  # [B, 55, 4, 4]

        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        if transform_mat_joint_1 is not None:
            transform_mat_joint = torch.matmul(
                transform_mat_joint_2, transform_mat_joint_1
            )  # [B, 55, 4, 4]
        else:
            transform_mat_joint = transform_mat_joint_2

        return transform_mat_joint, posed_joints

    def query_voxel_skinning_weights(self, vs):
        """using voxel-based skinning method
        vs: [B n c]
        """
        voxel_bbox = self.SMPLX_MODEL_voxel_bbox

        scale = voxel_bbox[..., 1] - voxel_bbox[..., 0]
        center = voxel_bbox.mean(dim=1)
        normalized_vs = (vs - center[None, None, :]) / scale[None, None]
        # mapping to [-1, 1] **3
        normalized_vs = normalized_vs * 2
        normalized_vs.to(self.SMPLX_MODEL_voxel_ws)

        B, N, _ = normalized_vs.shape

        query_ws = F.grid_sample(
            self.SMPLX_MODEL_voxel_ws.unsqueeze(0),  # 1 C D H W
            normalized_vs.reshape(1, 1, 1, -1, 3).to(self.SMPLX_MODEL_voxel_ws),
            align_corners=True,
            padding_mode="border",
        )
        query_ws = query_ws.view(B, -1, N)
        query_ws = query_ws.permute(0, 2, 1)

        return query_ws  # [B N C]
    
    def get_transform_mat_vertex(self, transform_mat_joint, query_points, fix_mask):
        batch_size = transform_mat_joint.shape[0]

        query_skinning = self.query_voxel_skinning_weights(query_points)
        skinning_weight = self.SMPLX_MODEL_skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)



        # query_skinning[fix_mask] = skinning_weight[fix_mask]

        query_skinning = torch.where(fix_mask.unsqueeze(-1).to(query_skinning.device), skinning_weight.to(query_skinning.device), query_skinning)




        transform_mat_vertex = torch.matmul(
            skinning_weight,
            transform_mat_joint.view(batch_size, 55, 16),
        ).view(batch_size, 40000, 4, 4)
        return transform_mat_vertex


    def lbs(self, xyz, transform_mat_vertex, trans):
        batch_size = xyz.shape[0]
        xyz = torch.cat(
            (xyz, torch.ones_like(xyz[:, :, :1])), dim=-1
        )  # 大 pose. xyz1 [B, N, 4]
        xyz = torch.matmul(transform_mat_vertex, xyz[:, :, :, None]).view(
            batch_size, 40000, 4
        )[
            :, :, :3
        ]  # [B, N, 3]
        if trans is not None:
            xyz = xyz + trans.unsqueeze(1)
        return xyz

    def transform_to_posed_verts_from_neutral_pose(
        self, mean_3d, smplx_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """

        batch_size = mean_3d.shape[0]
        shape_param = smplx_data["betas"]

        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )

        # smplx facial expression offset

        try:
            smplx_expr_offset = (
                smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.SMPLX_MODEL_expr_dirs
            ).sum(
                -1
            )  # [B, 1, 1, 50] x [N_V, 3, 50] -> [B, N_v, 3]
        except:
            assert False
            print("no use flame params")
            smplx_expr_offset = 0.0

        mean_3d = mean_3d + smplx_expr_offset  # 大 pose

        # get nearest vertex

        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        mask = (
            ((self.SMPLX_MODEL_is_rhand + self.SMPLX_MODEL_is_lhand + self.SMPLX_MODEL_is_face) > 0)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        # compute vertices-LBS function
        transform_mat_null_vertex = self.get_transform_mat_vertex(
            transform_mat_neutral_pose, mean_3d, mask
        )

        null_mean_3d = self.lbs(
            mean_3d, transform_mat_null_vertex, torch.zeros_like(smplx_data["trans"])
        )  # posed with smplx_param

        # blend_shape offset
        blend_shape_offset = blend_shapes(shape_param, self.SMPLX_MODEL_shape_dirs)
        null_mean3d_blendshape = null_mean_3d + blend_shape_offset


        # print(shape_param.shape)

        # assert False
        # get transformation matrix of the nearest vertex and perform lbs
        # joint_null_pose = self.get_zero_pose_human(
        #     shape_param=shape_param,  # target shape
        #     device=device,
        # )
        joint_null_pose = self.joint_null_pose

        # NOTE that the question "joint_zero_pose" is different with (transform_mat_neutral_pose)'s joints.
        transform_mat_joint, j3d = self.get_transform_mat_joint(
            None, joint_null_pose, smplx_data
        )

        # compute vertices-LBS function
        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint, mean_3d, mask
        )

        posed_mean_3d = self.lbs(
            null_mean3d_blendshape, transform_mat_vertex, smplx_data["trans"]
        )  # posed with smplx_param

        # as we do not use transform port [...,:,3],so we simply compute chain matrix
        neutral_to_posed_vertex = torch.matmul(
            transform_mat_vertex, transform_mat_null_vertex
        )  # [B, N, 4, 4]

        return posed_mean_3d, neutral_to_posed_vertex



    def animate_gs_model(
        self, offset_xyz, shs, opacity, scaling, rotation, query_points, smplx_data
    ):
        """
        query_points: [N, 3]
        """

        device = offset_xyz.device

        # build cano_dependent_pose
        cano_smplx_data_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
        ]

        merge_smplx_data = dict()
        for cano_smplx_data_key in cano_smplx_data_keys:
            warp_data = smplx_data[cano_smplx_data_key]
            cano_pose = torch.zeros_like(warp_data[:1])

            if cano_smplx_data_key == "body_pose":
                # A-posed
                cano_pose[0, 15, -1] = -torch.pi / 6
                cano_pose[0, 16, -1] = +torch.pi / 6

            merge_pose = torch.cat([warp_data, cano_pose], dim=0)
            merge_smplx_data[cano_smplx_data_key] = merge_pose 

        merge_smplx_data["betas"] = smplx_data["betas"]
        merge_smplx_data["transform_mat_neutral_pose"] = smplx_data[
            "transform_mat_neutral_pose"
        ]
        if True:
        # with torch.autocast(device_type=device.type, dtype=torch.float32):
            mean_3d = (
                query_points + offset_xyz
            )  # [N, 3]  # canonical space offset.

            # matrix to warp predefined pose to zero-pose
            transform_mat_neutral_pose = merge_smplx_data[
                "transform_mat_neutral_pose"
            ]  # [55, 4, 4]
            num_view = merge_smplx_data["body_pose"].shape[0]  # [Nv, 21, 3]
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
            query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1)
            transform_mat_neutral_pose = transform_mat_neutral_pose.unsqueeze(0).repeat(
                num_view, 1, 1, 1
            )

            mean_3d, transform_matrix = (
                self.transform_to_posed_verts_from_neutral_pose(
                    mean_3d ,
                    merge_smplx_data,
                    query_points,
                    transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
                    device=device,
                )
            )  # [B, N, 3]

            # rotation appearance from canonical space to view_posed
            # num_view, N, _, _ = transform_matrix.shape
            # transform_rotation = transform_matrix[:, :, :3, :3]

            # rigid_rotation_matrix = torch.nn.functional.normalize(
            #     matrix_to_quaternion(transform_rotation), dim=-1
            # )  + debug_time * 0.0


            num_view, N, _, _ = transform_matrix.shape
            transform_rotation = transform_matrix[..., :3, :3]  # 等价但更 ONNX 友好 

            q = matrix_to_quaternion_onnxsafe(transform_rotation)  # 新函数
            rigid_rotation_matrix = torch.nn.functional.normalize(q, dim=-1)


            # I = matrix_to_quaternion(torch.eye(3)).to(device)
            # # inference constrain
            # is_constrain_body = self.SMPLX_MODEL_is_constrain_body
            # rigid_rotation_matrix[:, is_constrain_body] = I
            # scaling[is_constrain_body] = scaling[
            #     is_constrain_body
            # ].clamp(max=0.02)


            # 构造 (num_view, N, 4) 的单位四元数张量
            # I_quat = matrix_to_quaternion_onnxsafe(torch.eye(3, device=device)).view(1, 1, 4) 
            # I_full = I_quat.expand(rigid_rotation_matrix.shape)
            # I_full 形状与 rigid_rotation_matrix 一致，且等于 [1,0,0,0]
            I_full = torch.cat([
                torch.ones_like(rigid_rotation_matrix[..., :1]),
                torch.zeros_like(rigid_rotation_matrix[..., 1:])
            ], dim=-1)



            # 掩码广播到 (num_view, N, 4)
            mask_body = self.SMPLX_MODEL_is_constrain_body.bool().unsqueeze(0).unsqueeze(-1)
            rigid_rotation_matrix = torch.where(mask_body.to(rigid_rotation_matrix.device), I_full.to(rigid_rotation_matrix.device), rigid_rotation_matrix)
            # scaling 限幅（若 scaling 是 (N,3)，用 (N,1)；若是 (num_view,N,3)，用 (1,N,1)）
            mask_scale = self.SMPLX_MODEL_is_constrain_body.bool().unsqueeze(-1)

            scaling = scaling
            scaling = torch.where(mask_scale.to(scaling.device), torch.clamp(scaling, max=0.02).to(scaling.device), scaling)





            rotation_neutral_pose = rotation.unsqueeze(0).repeat(num_view, 1, 1)

            # TODO do not move underarm gs

            # QUATERNION MULTIPLY
            rotation_pose_verts = quaternion_multiply(
                rigid_rotation_matrix, rotation_neutral_pose
            )
            # rotation_pose_verts = rotation_neutral_pose
        
        gaussian_xyz = mean_3d[0]
        canonical_xyz = mean_3d[1]
        gaussian_opacity = opacity # copy
        gaussian_rotation = rotation_pose_verts[0]
        canonical_rotation = rotation_pose_verts[1]
        gaussian_scaling = scaling
        gaussian_rgb = shs # copy

        return gaussian_xyz, canonical_xyz, gaussian_rgb, gaussian_opacity, gaussian_rotation, canonical_rotation, gaussian_scaling, rigid_rotation_matrix




    def forward(self, t_orig):





        # esti_shape = self.esti_shape
        gaussian_canon_dxdydz = self.dxdydz.to(t_orig.device)
        query_points = self.positions.to(t_orig.device)
        gaussian_canon_rgb = self.colors.to(t_orig.device)
        gaussian_canon_opacity = self.opacities.to(t_orig.device)
        gaussian_canon_scaling = self.scales.to(t_orig.device)
        gaussian_canon_rotation = self.rotations.to(t_orig.device)
        transform_mat_neutral_pose = self.transform_mat_neutral_pose.to(t_orig.device)



        # select_index = 0
        # print('--------------------self.smplx_poses')
        # print(self.smplx_poses.shape)
        # print(self.smplx_poses[select_index].shape)
        # print(self.smplx_shapes.shape)
        # print(self.smplx_R.shape)
        # print(self.smplx_Th.shape)
        global debug_time
        debug_time = t_orig
        t = torch.tensor(0.0)
        t = debug_time
        t = (t ) % 1.0
        # t = t * 0.0

        T = self.smplx_betas.shape[0]  # 81

        idx_f = t * (T - 1) 

        # poses  = _pick_one_hot(self.smplx_poses, idx_f)[None, :]          # (1,72)
        # shapes = _pick_one_hot(self.smplx_shapes, idx_f)[None, :]         # (1,10)
        # R_flat = _pick_one_hot(self.smplx_R.reshape(T, 9), idx_f)         # (9,)
        # R      = R_flat.view(3, 3)                                        # (3,3)
        # Th     = _pick_one_hot(self.smplx_Th, idx_f)[None, :]             # (1,3)


        smplx_data = {
            'betas': pick_with_gather(self.smplx_betas, idx_f).unsqueeze(0) , 
            'root_pose': pick_with_gather(self.smplx_root_pose, idx_f).unsqueeze(0), 
            'body_pose': pick_with_gather(self.smplx_body_pose, idx_f).unsqueeze(0), 
            'jaw_pose': pick_with_gather(self.smplx_jaw_pose, idx_f).unsqueeze(0), 
            'leye_pose': pick_with_gather(self.smplx_leye_pose, idx_f).unsqueeze(0), 
            'reye_pose': pick_with_gather(self.smplx_reye_pose, idx_f).unsqueeze(0), 
            'lhand_pose': pick_with_gather(self.smplx_lhand_pose, idx_f).unsqueeze(0), 
            'rhand_pose': pick_with_gather(self.smplx_rhand_pose, idx_f).unsqueeze(0), 
            'trans': pick_with_gather(self.smplx_trans, idx_f).unsqueeze(0), 
            'expr': pick_with_gather(self.smplx_expr, idx_f).unsqueeze(0), 
            'transform_mat_neutral_pose': transform_mat_neutral_pose, 
        }


        gaussian_xyz, canonical_xyz, gaussian_rgb, \
            gaussian_opacity, gaussian_rotation, canonical_rotation, \
                gaussian_scaling, transform_matrix = self.animate_gs_model(
                    gaussian_canon_dxdydz, gaussian_canon_rgb , gaussian_canon_opacity, 
                    gaussian_canon_scaling, gaussian_canon_rotation, 
                    query_points, smplx_data
                )


        print(gaussian_xyz.shape)
        bbox_min = gaussian_xyz.amin(dim=0, keepdim=True)   # [1,3]
        bbox_max = gaussian_xyz.amax(dim=0, keepdim=True)   # [1,3]
        center = (bbox_min + bbox_max) * 0.5                # [1,3]

        # 继续沿用你原来的 numpy 往返
        center = center.detach().cpu().numpy()              # 注意：cuda 上必须先 cpu()
        center = torch.from_numpy(center).to(gaussian_xyz.device).to(gaussian_xyz.dtype)

        gaussian_xyz = gaussian_xyz - center

        # safe to create graph here
        cov3D_precomp = get_covariance(
            gaussian_scaling , 
            gaussian_rotation 
        )
        # safe to create graph here
        packed = compress_gauhuman_gaussians_torch(
            positions=gaussian_xyz,
            cov3D_precomp=cov3D_precomp  , 
            opacity=gaussian_opacity ,
            shs=gaussian_rgb# + t_orig * 0.0
        )

        gaussian_f16 = packed["gaussian_f16"]  # (N,10)  f16
        # sh_f16       = packed["sh_f16"]        # (N,48)  f16

        # z_g = (gaussian_f16[:1, :1] - gaussian_f16[:1, :1]).reshape(())  # scalar 0, dtype跟随下行
        # z_g = z_g.to(gaussian_f16.dtype)

        # z_s = (sh_f16[:1, :1] - sh_f16[:1, :1]).reshape(())
        # z_s = z_s.to(sh_f16.dtype)

        # gauss_tail = z_g.expand(MAX_N, gaussian_f16.shape[1])  # (MAX_N,10)
        # sh_tail    = z_s.expand(MAX_N, sh_f16.shape[1])        # (MAX_N,48)

        # gauss_fixed = torch.cat([gaussian_f16, gauss_tail], dim=0)[:MAX_N, :]
        # sh_fixed    = torch.cat([sh_f16,       sh_tail],    dim=0)[:MAX_N, :]
        # sh_fixed    = F.pad(sh_f16, (0, 1), value=0)  # 在最后一维右侧补 1 个元素
        
        # print(sh_f16.shape)
        num_points = torch.tensor([gaussian_f16.shape[0]], dtype=torch.int32)
        return (gaussian_f16, self.colors_4, num_points)






def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:

    return quat_to_rotmat(quaternions)
    # """
    # Convert rotations given as quaternions to rotation matrices.

    # Args:
    #     quaternions: quaternions with real part first,
    #         as tensor of shape (..., 4).

    # Returns:
    #     Rotation matrices as tensor of shape (..., 3, 3).
    # """
    # r, i, j, k = torch.unbind(quaternions, -1)
    # # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    # two_s = 2.0 / (quaternions * quaternions).sum(-1)

    # o = torch.stack(
    #     (
    #         1 - two_s * (j * j + k * k),
    #         two_s * (i * j - k * r),
    #         two_s * (i * k + j * r),
    #         two_s * (i * j + k * r),
    #         1 - two_s * (i * i + k * k),
    #         two_s * (j * k - i * r),
    #         two_s * (i * k - j * r),
    #         two_s * (j * k + i * r),
    #         1 - two_s * (i * i + j * j),
    #     ),
    #     -1,
    # )
    # return o.reshape(quaternions.shape[:-1] + (3, 3))






def sinc_normalized_onnx(x: torch.Tensor) -> torch.Tensor:
    """
    ONNX-friendly 实现：sin(pi*x)/(pi*x)，在 x==0 时返回 1。
    等价于 torch.sinc(x)。
    """
    pi = torch.tensor(torch.pi, dtype=x.dtype, device=x.device)
    y = x * pi
    return torch.where(x == 0, torch.ones_like(x), torch.sin(y) / y)



def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * sinc_normalized_onnx(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
    )







def axis_angle_to_matrix(axis_angle: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if not fast:
        return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)

    row0 = torch.stack([zeros, -rz,  ry], dim=-1)
    row1 = torch.stack([   rz, zeros, -rx], dim=-1)
    row2 = torch.stack([-ry,   rx,  zeros], dim=-1)
    cross_product_matrix = torch.stack([row0, row1, row2], dim=-2)  # (...,3,3)

    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + sinc_normalized_onnx(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )




def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)



def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)



def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)



@torch.no_grad()
def matrix_to_quaternion_onnxsafe(M: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    ONNX/WebGPU 友好版：只在最后一次 stack，避免多层嵌套 stack/concat。
    输入: M (..., 3, 3)
    输出: (..., 4) [w, x, y, z]，标准化且实部非负
    """
    if M.size(-1) != 3 or M.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {M.shape}.")

    # 直接按索引取元素，避免 reshape+unbind
    m00 = M[..., 0, 0]; m01 = M[..., 0, 1]; m02 = M[..., 0, 2]
    m10 = M[..., 1, 0]; m11 = M[..., 1, 1]; m12 = M[..., 1, 2]
    m20 = M[..., 2, 0]; m21 = M[..., 2, 1]; m22 = M[..., 2, 2]

    # 分支无栈化公式（Shepperd 变体），数值稳定 & 矢量化
    t0 = 1.0 + m00 + m11 + m22
    t1 = 1.0 + m00 - m11 - m22
    t2 = 1.0 - m00 + m11 - m22
    t3 = 1.0 - m00 - m11 + m22

    def _sqrt_pos(x):  # 等价于 sqrt(max(0,x)) 且导数安全
        z = torch.zeros_like(x)
        pos = x > 0
        if torch.is_grad_enabled():
            z[pos] = torch.sqrt(x[pos])
        else:
            z = torch.where(pos, torch.sqrt(x), z)
        return z

    s1 = m21 - m12
    s2 = m02 - m20
    s3 = m10 - m01
    c1 = s1 / (s1.abs() + eps)
    c2 = s2 / (s2.abs() + eps)
    c3 = s3 / (s3.abs() + eps)

    qw = 0.5 * _sqrt_pos(t0)
    qx = 0.5 * _sqrt_pos(t1) * c1
    qy = 0.5 * _sqrt_pos(t2) * c2
    qz = 0.5 * _sqrt_pos(t3) * c3

    q = torch.stack((qw, qx, qy, qz), dim=-1)
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=eps)
    q = torch.where(q[..., :1] < 0, -q, q)
    return q


def matrix_to_quaternion_indentity(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)



def visualize(t, gaussian_xyz, canonical_xyz, gaussian_rgb, gaussian_opacity, gaussian_scaling, gaussian_rotation, canonical_rotation, FoVx, FoVy, image_height, image_width, world_view_transform, full_proj_transform, camera_center):
    
    import math

    # FoVx, FoVy = 0.9069482352715645, 0.9016572363354971
    # image_height = 512
    # image_width = 512
    # active_sh_degree = 3
    # bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    # scaling_modifier = 1.0

    # debug = False

    # # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(positions, dtype=positions.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    # raster_settings = GaussianRasterizationSettings(
    #     image_height=image_height,
    #     image_width=image_width,
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
    #     scale_modifier=1.,
    #     viewmatrix=world_view_transform,
    #     projmatrix=full_proj_transform,
    #     sh_degree=0,
    #     campos=camera_center, 
    #     prefiltered=False,
    #     debug=False
    # )

    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    # print(positions.shape, screenspace_points.shape, colors.shape, opacity.shape, scales.shape, rotations.shape)

    # print(scales.shape, rotations.shape, transforms.shape)


    # cov3D_precomp = get_covariance(scales, rotations, scaling_modifier, transforms.squeeze())
    # scales, rotations = update_scale_rot_with_transform(scales, rotations, transforms.squeeze())

    # print(cov3D_precomp.shape)

    # print(positions.mean(0))

    # rendered_image, radii, depth, alpha = rasterizer(
    #     means3D = positions,
    #     means2D = screenspace_points,
    #     # shs = sh,
    #     colors_precomp = colors,
    #     opacities = opacity,
    #     # scales = scales,
    #     # rotations = rotations,
    #     cov3D_precomp = cov3D_precomp
    # )


    rendered_image, radii, depth, alpha = rasterizer(
        # means3D = canonical_xyz, 
        means3D = gaussian_xyz, 
        means2D = torch.zeros_like(canonical_xyz, dtype=canonical_xyz.dtype, requires_grad=False, device="cuda"), 
        shs = None, 
        colors_precomp = gaussian_rgb, 
        opacities = gaussian_opacity, 
        scales = gaussian_scaling, 
        # rotations = canonical_rotation, 
        rotations = gaussian_rotation, 
        cov3D_precomp = None
    )


    print('rendered_image shape:', rendered_image.shape)


    Image.fromarray((rendered_image.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)).save(f'debug_render_lhm_{t}.png')


def get_covariance(scales, rotations, transform=None):
    return build_covariance_from_scaling_rotation(scales, rotations, transform)




def build_covariance_from_scaling_rotation(scaling, rotation, transform):
    L = build_scaling_rotation(scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    if transform is not None:
        actual_covariance = transform @ actual_covariance
        actual_covariance = actual_covariance @ transform.transpose(1, 2)
    symm = strip_symmetric(actual_covariance )
    return symm




def strip_symmetric(C):
    # C: (N,3,3)
    xx = C[:, 0, 0]; xy = C[:, 0, 1]; xz = C[:, 0, 2]
    yy = C[:, 1, 1]; yz = C[:, 1, 2]; zz = C[:, 2, 2]
    return torch.stack([xx, xy, xz, yy, yz, zz], dim=-1)  # (N,6)






def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # s: (N,3)，r: (N,4) [w,x,y,z]
    R = quat_to_rotmat(r)                     # (N,3,3)
    R = R.to(s.dtype)                         # 对齐 dtype
    return R * s.to(R.dtype).unsqueeze(-2)    # 等价于 R @ diag(s)



def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3),device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def export_onnx(pth_path: Path, out_path: Path, motion_json, opset: int = 17):
    # data = load_gauhuman_ply(ply_path)
    data = load_LHM_pth(pth_path)
    model = GaussianSetModule(data, motion_json).eval().cpu()

    # params = torch.load('params.pth', map_location='cpu')

    # poses = params['poses'].cuda()  # torch.Size([1, 72])
    # shapes = params['shapes'].cuda()  # torch.Size([1, 10])
    # R = params['R'].cuda()  # torch.Size([3, 3])
    # Th = params['Th'].cuda()  # torch.Size([1, 3])

    dummy_t = torch.tensor(0.0).float().cpu()

    input_names = ['time_emb']
    output_names = ['gaussian_f16', 'color_rgb', 'num_points'] 
    dynamic_axes = {
        # 'camera': {0: 'batch'},
        # 'time':   {0: 'batch'},
        # 'gaussian_f16': {0: 'num_points', 1: 'gauss_fields'},  # (N,10)
        # 'sh_f16':       {0: 'num_points', 1: 'sh_packed'},     # (N,48)
    }

    # for t in range(80):
    #     model(t/80)

    # assert False
    if os.path.dirname(str(out_path))!='':
        os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_t),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"✅ Exported ONNX to: {out_path.resolve()}")



def vi_shape(vi):
    tt = vi.type.tensor_type
    if not tt.HasField("shape"): return []
    out = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            out.append(str(d.dim_param))  # 符号维 / 未知
        else:
            out.append("?")
    return out

def dump_ios(m, title):
    print(f"\n=== {title} ===")
    print("Outputs:")
    for o in m.graph.output:
        print(f"  {o.name}: {vi_shape(o)}")
    print("Inputs:")
    for i in m.graph.input:
        print(f"  {i.name}: {vi_shape(i)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, required=True)
    parser.add_argument("--motion_json", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本 (建议 17+)")
    args = parser.parse_args()

    pth_path = Path(args.pth)
    out_path = Path(args.out)
    motion_json = args.motion_json
    if not pth_path.exists():
        raise FileNotFoundError(f"未找到 pth: {pth_path}")

    export_onnx(pth_path, out_path, motion_json, opset=args.opset)

    fix_output_name(str(out_path), out_index=1, desired_name="color_rgb")

    # assert False



    import torch, onnx
    from onnx import shape_inference

    m = onnx.load(out_path)
    # for i, output in enumerate(m.graph.output):
    #         if i == 1:  # 找到第二个输出
    #             print(f"⚠️ 检测到名称丢失，正在修复: {output.name} -> color_rgb")
    #             output.name = "color_rgb"


    

    m = shape_inference.infer_shapes(m)

    
    onnx.save(m, out_path)

    dump_ios(m, "After  infer_shapes")


    produced = set()
    for n in m.graph.node:
        produced |= set(n.output)
    produced |= {i.name for i in m.graph.initializer}
    produced |= {i.name for i in m.graph.input}

    print("Graph outputs:", [o.name for o in m.graph.output])
    print("Missing outputs:", [o.name for o in m.graph.output if o.name not in produced])


    # 4) 简单断言（按你的预期改，比如最后一维=10）
    #    如果仍然是符号维（字符串），这里会跳过断言
    for o in m.graph.output:
        shp = vi_shape(o)
        print(shp)
        # if shp and isinstance(shp[-1], int):
        #     global MAX_N
        #     MAX_N = shp[0] 
        #     print(MAX_N)
        #     break
            #assert shp[-1] in (9, 10,48,1), f"{o.name} last dim {shp[-1]} not 9/10"

    # import onnxruntime as ort

    # so = ort.SessionOptions()
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # so.optimized_model_filepath = str(out_path)
    # # so.enableGraphCapture = True
    # # 用 CPU provider 也能生成优化图（重点是保存 optimized_model_filepath）
    # ort.InferenceSession(str(out_path), so, enableGraphCapture = True,providers=["CPUExecutionProvider"])
    # print("saved -> model.opt.onnx")

    # export_onnx(ply_path, out_path, opset=args.opset)

    # m = onnx.load(out_path)
    # m = shape_inference.infer_shapes(m)


if __name__ == "__main__":
    main()
