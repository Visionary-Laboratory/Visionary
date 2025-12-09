import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData
from typing import Dict, Optional

from loadply import load_3dgs_from_ply


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
    op  = torch.sigmoid(opacity.reshape(-1, 1).float())   # (N,1)
    scales_lin = torch.exp(scales_log.float())             # (N,3) 与前端一致
    cov6 = build_cov_from_scales_quat(scales_lin, rotations.float())  # (N,6)

    gauss_f32 = torch.cat([pos, op, cov6], dim=1)         # (N,10) f32
    return gauss_f32.to(torch.float16)                    # (N,10) f16


@torch.no_grad()
def pack_sh_to_48_f16(colors: torch.Tensor) -> torch.Tensor:
    """
    输入 colors: (N, K) f32，K >= 0
    目标: (N, 48) f16，布局:
      [R_dc, G_dc, B_dc,  (R1,G1,B1), (R2,G2,B2), ..., (R15,G15,B15)]
    约定: 你的 colors 顺序是 [R_dc, G_dc, B_dc, R_rest(<=15), G_rest(<=15), B_rest(<=15), ...]
         若 K 不足则自动补 0；若 K 超过则按上述规则裁到 48。
    """
    device = colors.device
    colors = colors.float()
    N, K = colors.shape

    out = torch.zeros((N, 48), dtype=torch.float16, device=device)

    # 1) DC 3 项（若 K<3，切片会是空，赋值安全）
    out[:, 0:3] = colors[:, 0:3].to(torch.float16)

    # 2) 每通道剩余项个数（不含 DC），但最大只取 15 项（SH3）
    #    这里不用 if，直接切片到安全范围
    #    R: [3 : 3+nr), G: [3+nr : 3+2nr), B: [3+2nr : 3+3nr)
    #    先按理论长度算 nr，然后各自再裁 15
    nr_full = (K - 3) // 3                      # 可能为 0
    nr = min(15, int(nr_full))                  # Python 用于切片上限，不涉及张量比较

    if nr > 0:
        r_rest = colors[:, 3:3+nr]                              # (N, nr)
        g_rest = colors[:, 3+nr:3+2*nr]                         # (N, nr)
        b_rest = colors[:, 3+2*nr:3+3*nr]                       # (N, nr)
        interleaved = torch.stack([r_rest, g_rest, b_rest], dim=2).reshape(N, nr*3)  # (N, 3*nr)
        out[:, 3:3+nr*3] = interleaved.to(torch.float16)

    return out


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


    gaussian_f16 = pack_gaussian_f16(positions, scales, rotations, opacity)
    sh_f16 = pack_sh_to_48_f16(colors)

    return {"gaussian_f16": gaussian_f16, "sh_f16": sh_f16}


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
        packed = compress_gaussians_torch(
            positions=self.positions,
            scales=getattr(self, "scales", None),
            rotations=getattr(self, "rotations", None),
            opacity=getattr(self, "opacity", None),
            colors=getattr(self, "colors", None),
        )
        num_points = torch.tensor([self.positions.shape[0]], dtype=torch.int32)
        return (packed["gaussian_f16"], packed["sh_f16"], num_points)


def export_onnx(ply_path: Path, out_path: Path, opset: int = 17):
    data = load_3dgs_from_ply(ply_path)
    model = GaussianSetModule(data).eval()

    dummy_camera = torch.zeros(1, 16, dtype=torch.float32)
    dummy_time = torch.zeros(1, 1, dtype=torch.float32)

    input_names = ['camera', 'time']
    output_names = ['gaussian_f16', 'sh_f16', 'num_points'] 
    dynamic_axes = {
        # 'camera': {0: 'batch'},
        # 'time':   {0: 'batch'},
        # 'gaussian_f16': {0: 'num_points', 1: 'gauss_fields'},  # (N,10)
        # 'sh_f16':       {0: 'num_points', 1: 'sh_packed'},     # (N,48)
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
