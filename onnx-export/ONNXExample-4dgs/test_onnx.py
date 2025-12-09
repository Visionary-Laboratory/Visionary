import sys, os, time
import numpy as np

# 选择正确的包名（Apple Silicon 也叫 onnxruntime）
try:
    import onnxruntime as ort
except ImportError:
    print("Please `pip install onnxruntime` (or onnxruntime-silicon / onnxruntime-gpu)")
    sys.exit(1)

MODEL = os.environ.get("MODEL", "gaussians4d.onnx")

# 1) 选择执行提供者（EP）
avail = ort.get_available_providers()
if "CUDAExecutionProvider" in avail:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]

print("Available providers:", avail)
print("Using providers    :", providers)

# 2) Session 选项（默认启用图优化）
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 3) 创建 Session
sess = ort.InferenceSession(MODEL, sess_options=so, providers=providers)
print("Session outputs:", [o.name for o in sess.get_outputs()])


# ==================================================

# 4) 构造输入（与你导出时的 dummy 一致）
# 实际上没有用到，所以被优化掉了
feeds = {
    "time":   np.zeros((1),  dtype=np.float32),
}

# 5) 运行并计时
t0 = time.time()
outputs = sess.run(None, feeds)   # None 表示取全部输出
dt = (time.time() - t0) * 1000
print(f"Inference done in {dt:.2f} ms")

# 6) 打印每个输出的形状/范围
for meta, arr in zip(sess.get_outputs(), outputs):
    name = meta.name
    print(f"- {name:10s} shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.6f} max={arr.max():.6f}")
# print(type(outputs))
# print(outputs[0][:10])
# print(outputs[1][:10])
# print(outputs[2][:10])
assert False
# 7) 一点健壮性检查：N 是否一致
shapes = { meta.name: arr.shape for meta, arr in zip(sess.get_outputs(), outputs) }
N = shapes["positions"][0]
assert shapes["scales"][0]    == N
assert shapes["rotations"][0] == N
assert shapes["colors"][0]    == N
assert shapes["opacity"][0]   == N
print(f"✅ Consistent N={N} across outputs.")
