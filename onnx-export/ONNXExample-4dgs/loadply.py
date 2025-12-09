import argparse 
from pathlib import Path 
import numpy as np 
import torch 
import torch.nn as nn 
from plyfile import PlyData

def _get_struct_field(arr, name):
    """安全读取结构化数组字段，如果不存在返回 None（不做类型转换）"""
    try:
        return arr[name]
    except (ValueError, KeyError):
        return None


def _try_stack(arr, names, dtype=None):
    """
    按给定字段名顺序拼接为 (N, K)。
    不做归一化、不改 dtype（除非显式传入 dtype）。
    """
    cols = []
    for n in names:
        col = _get_struct_field(arr, n)
        if col is None:
            return None
        cols.append(col.reshape(-1, 1))
    out = np.concatenate(cols, axis=1)
    if dtype is not None:
        out = out.astype(dtype)
    return out


def _concat_or_none(parts):
    """把若干 (N, K_i) 的数组按列拼接；若全是 None 返回 None；不做任何数值处理"""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    N = parts[0].shape[0]
    parts = [p.reshape(N, -1) for p in parts]
    return np.concatenate(parts, axis=1)


def _collect_series(arr, prefixes):
    """
    收集一组前缀列表中存在的 prefix_* 系列字段，每个前缀各自组装成一块 (N, Kp)。
    返回按传入前缀顺序收集到的数组列表（不存在的前缀跳过）。
    不做任何数值处理。
    """
    out = []
    for p in prefixes:
        m = _find_prefixed_series(arr, p)
        if m is not None:
            out.append(m)
    return out


def _find_prefixed_series(arr, prefix, count=None, alt_prefixes=()):
    """
    查找形如 prefix_0, prefix_1, ... 的字段并拼成 (N, K)。
    不做归一化/类型转换。
    """
    names = arr.dtype.names or ()
    cand_prefixes = (prefix,) + tuple(alt_prefixes)

    idxs = []
    for p in cand_prefixes:
        for n in names:
            if n.startswith(p + "_"):
                suffix = n[len(p) + 1:]
                if suffix.isdigit():
                    idxs.append((p, int(suffix)))
    if not idxs:
        return None

    K = (max(i for _, i in idxs) + 1) if count is None else count

    cols = []
    N = len(arr)
    for i in range(K):
        col = None
        for p in cand_prefixes:
            col = _get_struct_field(arr, f"{p}_{i}")
            if col is not None:
                break
        if col is None:
            return None
        cols.append(col.reshape(N, 1))
    return np.concatenate(cols, axis=1)


def _concat_or_none(parts):
    """把若干 (N, K_i) 的数组按列拼接；若全是 None 返回 None；不做任何数值处理"""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    N = parts[0].shape[0]
    parts = [p.reshape(N, -1) for p in parts]
    return np.concatenate(parts, axis=1)


def load_3dgs_from_ply(ply_path: Path, onlyrgb=False):
    """
    仅从 PLY 直接读出各字段；不做任何数值处理：
    - 不归一化四元数
    - 不乘 Y00
    - 不把 uint8 颜色缩放到 0..1
    - 不填默认值（缺失就返回 None）
    - 不做 NaN/Inf 清理
    """
    ply = PlyData.read(str(ply_path))
    if 'vertex' not in ply:
        raise ValueError("PLY 中缺少 'vertex' element。")
    v = ply['vertex'].data
    N = len(v)

    # positions（必须）：只拼接，不改 dtype
    pos = _try_stack(v, ['x', 'y', 'z'])
    pos = pos + (0.0,1.0,0.0)
    if pos is None:
        raise ValueError("PLY 中没有 x/y/z 字段，无法组装 positions。")

    # scales：先尝试 scale_0..2 / scales_0..2；不行再尝试 scale_x/y/z
    scales = _find_prefixed_series(v, 'scale', count=3, alt_prefixes=('scales',))
    if scales is None:
        scales = _try_stack(v, ['scale_x', 'scale_y', 'scale_z'])

    # rotations：rotation_0..3 / rot_0..3 / quat_0..3 / quaternion_0..3；不作归一化
    rot = _find_prefixed_series(v, 'rotation', count=4, alt_prefixes=('rot', 'quat', 'quaternion'))

    # colors：优先 f_dc / features_dc 的前三列；否则 r,g,b / red,green,blue；不乘 Y00、不缩放
    # 1) RGB
    
    rgb = _try_stack(v, ['r', 'g', 'b'])
    if rgb is None:
        rgb = _try_stack(v, ['red', 'green', 'blue'])

    # 2) DC 与 AC（可同时存在多个前缀，全部保留并拼接）
    dc_parts = _collect_series(v, ('f_dc', 'features_dc', 'sh_dc'))

    ac_parts = _collect_series(v, ('f_rest', 'features_rest', 'sh_rest'))
    if onlyrgb:
        # dc_parts = []
        ac_parts = []
    # 3) 最终拼接（哪个有就拼哪个；都没有则 None）
    colors = _concat_or_none(([rgb] if rgb is not None else []) + dc_parts + ac_parts)

    print(colors.shape)
    # opacity：存在则拉成 (N,1)，否则保持 None
    opacity = _get_struct_field(v, 'opacity')
    if opacity is not None:
        opacity = np.asarray(opacity).reshape(N, 1)

    return {
        'positions': pos,       # (N,3)，原始 dtype
        'scales': scales,       # (N,3) 或 None
        'rotations': rot,       # (N,4) 或 None
        'colors': colors,       # (N,3) 或 None
        'opacity': opacity,     # (N,1) 或 None
    }



def load_scaffold_gs_from_ply(ply_path: Path, onlyrgb=False):
    """
    从 PLY 文件直接读取 Scaffold-GS 的字段；不做额外数值处理。
    返回一个 dict，包含 anchor / offsets / features / scales / rotations / opacity。
    """
    ply = PlyData.read(str(ply_path))
    if 'vertex' not in ply:
        raise ValueError("PLY 中缺少 'vertex' element。")
    v = ply['vertex'].data
    N = len(v)

    # === anchors (x,y,z) ===
    anchor = np.stack([np.asarray(v["x"]),
                       np.asarray(v["y"]),
                       np.asarray(v["z"])], axis=1).astype(np.float32)

    # === opacity ===
    opacity = np.asarray(v["opacity"], dtype=np.float32).reshape(N, 1)

    # === scales ===
    scale_names = [p.name for p in ply['vertex'].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.stack([np.asarray(v[name], dtype=np.float32) for name in scale_names], axis=1)

    # === rotations ===
    rot_names = [p.name for p in ply['vertex'].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.stack([np.asarray(v[name], dtype=np.float32) for name in rot_names], axis=1)

    # === anchor_feats ===
    feat_names = [p.name for p in ply['vertex'].properties if p.name.startswith("f_anchor_feat")]
    feat_names = sorted(feat_names, key=lambda x: int(x.split('_')[-1]))
    anchor_feats = np.stack([np.asarray(v[name], dtype=np.float32) for name in feat_names], axis=1)

    # === offsets ===
    offset_names = [p.name for p in ply['vertex'].properties if p.name.startswith("f_offset")]
    offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
    offsets = np.stack([np.asarray(v[name], dtype=np.float32) for name in offset_names], axis=1)
    # reshape (N, 3*k) → (N, k, 3)
    offsets = offsets.reshape((N, 3, -1)).transpose(0, 2, 1)

    return {
        "anchors": anchor,          # (N,3)
        "scales": scales,           # (N,3)
        "rotations": rots,          # (N,4)
        "opacity": opacity,         # (N,1)
        "anchor_feats": anchor_feats, # (N,C)
        "offsets": offsets          # (N, n_offsets, 3)
    }

def load_4dgs_from_ply(ply_path: Path, onlyrgb=False):
    """
    https://github.com/hustvl/4DGaussians.git

    从 PLY 文件直接读取 4DGS 的字段；不做额外数值处理。
    返回一个 dict，包含 features / scales / rotations / opacity。
    """
    plydata = PlyData.read(ply_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return {
        'positions': xyz,       # (N, 3)，原始 dtype
        'scales': scales,       # (N, 3) 或 None
        'rotations': rots,       # (N, 4) 或 None
        'features_dc': features_dc,       # (N, 16, 3) 或 None
        'features_extra': features_extra,       # (N, 16, 3) 或 None
        'opacity': opacities,     # (N, 1) 或 None
    }
