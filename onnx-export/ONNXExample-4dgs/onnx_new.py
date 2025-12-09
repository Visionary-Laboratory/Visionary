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


def load_3dgs_from_ply(ply_path: Path):
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

    # 3) 最终拼接（哪个有就拼哪个；都没有则 None）
    colors = _concat_or_none(([rgb] if rgb is not None else []) + dc_parts + ac_parts)


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

class GaussianSetModule(nn.Module):
    """
    一个最小可用的“常量输出”模块。
    - 输入: camera(B,16) 和 time(B,1)，当前版本忽略
    - 输出: positions(N,3), scales(N,3), rotations(N,4), colors(N,3), opacity(N,1)
    """
    def __init__(self, ply_data: dict):
        super().__init__()
        # 注册为 buffer，这样导出 ONNX 时会作为 initializers 存在于模型里
        for k, np_arr in ply_data.items():
            t = torch.from_numpy(np_arr.astype(np.float32))
            self.register_buffer(k, t, persistent=True)

        # torch.nn.parameter(MLP,etc.)

    def forward(self, camera: torch.Tensor, time: torch.Tensor):
        # 先把 pipeline 走通：忽略输入，直接输出常量
        
        return (self.positions, self.scales, self.rotations, self.colors, self.opacity)

def export_onnx(ply_path: Path, out_path: Path, opset: int = 17):

    data = load_3dgs_from_ply(ply_path)
    model = GaussianSetModule(data).eval()

    # dummy 输入（保留接口，后面你要用 camera/time 做调制可以直接改 forward）
    dummy_camera = torch.zeros(1, 16, dtype=torch.float32)
    dummy_time = torch.zeros(1, 1, dtype=torch.float32)

    input_names = ['camera', 'time']
    output_names = ['positions', 'scales', 'rotations', 'colors', 'opacity']
    dynamic_axes = {
        # 允许未来按 batch 传 camera/time；当前不影响输出
        'camera': {0: 'batch'},
        'time': {0: 'batch'},
        # 输出目前固定为 (N, C)，如需做 batch 展开，可在 forward 里扩展维度
    }

    torch.onnx.export(
        model,
        (dummy_camera, dummy_time),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"✅ Exported ONNX to: {out_path.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default="test_gs.ply", help="3DGS PLY 文件路径")
    parser.add_argument("--out", type=str, default="gaussians3d.onnx", help="导出的 ONNX 路径")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本 (建议 17+)")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    out_path = Path(args.out)
    if not ply_path.exists():
        raise FileNotFoundError(f"未找到 PLY: {ply_path}")

    export_onnx(ply_path, out_path, opset=args.opset)

if __name__ == "__main__":
    main()
