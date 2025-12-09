# 🚀 Visionary: 数据准备与 ONNX 导出指南

<!-- 本统一指南涵盖了为 Visionary Viewer 准备、训练和导出的流程。 -->

Visionary 基于标准化的 **Gaussian Generator** 协议：只要你的 3DGS 系列算法（经典 / 结构化 / 4DGS、Avatar，或任意自定义变体）能导出 ONNX 并输出逐帧的高斯属性（位置、尺度、旋转、颜色等），就可以在无需修改 WebGPU 渲染器或着色器的情况下接入查看器。本文档中的各个流水线（Avatar、4DGS、Scaffold-GS 等）可作为参考实现，你可以将它们当作模板把自己的方法适配到 Visionary 运行时。

为了让你的 Gaussian Generator 在 Visionary 上高效运行，推荐一些针对 WebGPU 运行时的 ONNX 导出实用技巧：

- **导出便于图捕获的模型。** 尽量避免动态控制流和高度动态的张量形状，使 ONNX Runtime WebGPU 能启用 graph capture。保持稳定图（固定 batch/序列形状、无 Python/Loop 风格算子、无奇异 dtype）能在捕获后显著提速。
- **遵循下文示例的索引模式。** 在切片或索引高斯属性（位置、尺度、旋转、颜色等）时，尽量复用文档中参考流水线的索引策略，保持内存布局连续并与 WebGPU kernel 及后处理工具兼容。
- **用手写实现替换内置 Norm 算子。** ONNX Runtime 的 WebGPU 后端在 Norm、LayerNormalization、RMSNorm 上存在已知问题和性能差异，建议导出前改写为基础算子组合（如 `ReduceMean` + `Sub` + `Mul` + `Add`），或导出后用预处理脚本替换。
- **避免巨型单个 `Concat` / `Split`。** WebGPU shader 受资源绑定数量限制，如果模型有非常大的 `Concat` 或 `Split`（大量输入/输出），请拆成多段 `Concat`/`Split` 再合并，可提升编译稳定性。

本统一指南涵盖了为 **Visionary Viewer** 准备、训练和导出数据的流程。它包括针对可动画化 Avatar、动态场景 (4DGS)、结构化静态场景 (Scaffold-GS) 和通用格式转换的说明。

## 📋 目录
1. [可动画化 Avatar (基于 SMPL-X)](#1-可动画化-avatar-onnx-模型)
2. [4D Gaussian Splatting (动态场景)](#2-4d-gaussian-splatting-4dgs-导出)
3. [Scaffold-GS (结构化静态场景)](#3-scaffold-gs-导出)
4. [通用 3DGS 格式转换](#4-通用-3dgs-格式转换工具)

---

## 1. 可动画化 Avatar ONNX 模型

使用此流程生成由 SMPL-X 动作数据驱动的可动画化 Avatar。

### **1.1 环境配置**

**创建并激活环境 (Python 3.10)**

```bash
conda create -n visionary_avatar python==3.10 -y
conda activate visionary_avatar
```

**安装核心框架**

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
pip install ninja psutil tb-nightly
```

**安装自定义依赖**

> **故障排除提示：** 如果在安装 **PyTorch3D** 或 **Gaussian Rasterization** 时遇到 `ModuleNotFoundError: No module named 'torch'` 错误，请在命令后附加 `--no-build-isolation` 标志。

```bash
# 核心 3D 库
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/hitsz-zuoqi/sam2/
pip install git+https://github.com/XPixelGroup/BasicSR
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/

# 其他实用工具
pip install opencv-python roma smplx tqdm scikit-image huggingface_hub[cli] modelscope kornia timm accelerate diffusers==0.32.0 plyfile trimesh matplotlib jaxtyping decord transformers==4.46.2 sentencepiece chumpy gfpgan xfuser onnxruntime-gpu onnx natsort

# 降级 Numpy 以确保兼容性
pip install numpy==1.23.5
```

### **1.2 模型权重与资源**

从 HuggingFace 下载预训练模型和 GFPGAN 权重。

```bash
# 下载仓库
hf download MyNiuuu/Visionary_avatar --local-dir ./Visionary_avatar --local-dir-use-symlinks False

# 整理文件
mv ./Visionary_avatar/pretrained_models .
mv ./Visionary_avatar/gfpgan .

# 清理
rm -rf ./Visionary_avatar
```

### **1.3 动作数据准备 (AMASS-CMU)**
从 AMASS 数据集获取 SMPL-X 动作序列以驱动 Avatar。

1.  **注册/登录：** 访问 [AMASS 网站](https://amass.is.tue.mpg.de) 并登录。
2.  **导航至下载：** 前往 [下载页面](https://amass.is.tue.mpg.de/download.php)。
3.  **选择数据集：** 找到 **CMU** 数据集。
4.  **下载：** 点击 `SMPL-X N` 下载 zip 文件。
5.  **解压：** 将内容解压到 `./motions` 目录。

### **1.4 生成 ONNX Avatar 模型**

**执行运行脚本：**

```bash
bash run.sh
```
---

## 2. 4D Gaussian Splatting (4DGS) 导出

用于运行 4DGS 项目并将训练好的动态场景表示导出为 ONNX 格式的流程。

### **2.1 环境配置**

```bash
git clone https://github.com/hustvl/4DGaussians
cd 4DGaussians
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7 
conda activate Gaussians4D
    
# 安装依赖
pip install -r requirements.txt
pip install onnx

# 安装子模块
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

### **2.2 代码准备 (关键步骤)**

要导出为 ONNX，必须修改 4DGaussians 仓库中的 `train.py` 以保存 hex-plane AABB。

**修改第 299-313 行左右：**

*更改前：*
```python
tb_writer = prepare_output_and_logger(expname)
gaussians = GaussianModel(dataset.sh_degree, hyper)
dataset.model_path = args.model_path
timer = Timer()
scene = Scene(dataset, gaussians, load_coarse=None)
```

*更改后：*
```python
args.model_path = os.path.join("./output/", expname)
os.makedirs(args.model_path, exist_ok = True)
gaussians = GaussianModel(dataset.sh_degree, hyper)
dataset.model_path = args.model_path
timer = Timer()
scene = Scene(dataset, gaussians, load_coarse=None)
# 为 ONNX 导出添加：
grid_aabb = scene.gaussians._deformation.deformation_net.get_aabb
args.grid_aabb = [x.cpu().tolist() for x in grid_aabb]
tb_writer = prepare_output_and_logger(expname)
```

*为确保路径一致，请**删除**随后的自动 `args.model_path` 生成逻辑（UUID 生成）：*
```python
if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname
    
        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
```

### **2.3 数据准备**

*   **合成场景：** 使用 [D-NeRF 数据集](https://github.com/albertpumarola/D-NeRF)。您可以从 [dropbox](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0) 下载数据集。

*   **真实场景：** 使用 [Neural 3D Video 数据集](https://github.com/facebookresearch/Neural_3D_Video)。为了节省内存，请提取每个视频的帧并应用 [COLMAP](https://colmap.github.io/) 获取初始点云。

    1.  **提取帧：**
        ```bash
        python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
        ```

    2.  **生成点云：**
        ```bash
        bash colmap.sh data/dynerf/cut_roasted_beef llff
        ```

    3.  **下采样点云：**
        ```bash
        python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
        ```

**目录结构**
最终数据集应按如下方式组织：

```text
├── data
│   | dnerf 
│     ├── hook
│     ├── standup 
│     ├── ...
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│       │ points3D_downsample2.ply
│       │ poses_bounds.npy
│     ├── cut_roasted_beef
│     ├── ...
```

### **2.4 训练**

示例训练命令 (D-NeRF `hook` 场景)：

```bash
python train.py -s data/dnerf/hook --port 6017 --expname "dnerf/hook" --configs arguments/dnerf/hook.py 
```
训练后，检查点和输出保存在 `./output/dnerf/hook` 中，如下所示：
```text
├── output
│   | dnerf 
│     ├── hook
│        ├── point_cloud
│           ├── iteration_14000
│              ├── deformation.pth
│              ├── deformation_accum.pth
│              ├── deformation_table.pth
│              ├── point_cloud.ply
│        | cfg_args
│     ├── standup 
│     ├── ...
│   | dynerf
│     ├── cook_spinach
│     ├── ...
```

### **2.5 导出 ONNX**

在 4D-GS 环境中，使用导出脚本（确保您位于 `ONNXExample` 目录结构中）：

```bash
git clone -b 4dgs https://github.com/Visionary-Laboratory/ONNXExample.git
cd ONNXExample

python onnx_template.py --ply path/to/output/dnerf/hook/point_cloud/iteration_14000/point_cloud.ply \
                  --out your/prefered/onnxpath/gaussians4d.onnx
```
---

## 3. Scaffold-GS 导出

运行 Scaffold-GS 并将训练好的静态场景导出为 ONNX 的流程。

### **3.1 环境配置**

```bash
git clone https://github.com/city-super/Scaffold-GS.git --recursive
cd Scaffold-GS

# 仅限 Windows: SET DISTUTILS_USE_SDK=1 
conda env create --file environment.yml
conda activate scaffold_gs
pip install onnx
```

### **3.2 数据准备**

创建一个 `data/` 文件夹。数据应遵循标准的 Colmap 结构：

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

**公开数据**
您可以下载标准数据集并将它们解压到 `data/` 文件夹中：

*   **BungeeNeRF:** 可在 [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing) 或 [百度网盘 (提取码: 4whv)](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA) 下载。
*   **MipNeRF360:** 由论文作者提供 [在此处](https://jonbarron.info/mipnerf360/)。我们测试的场景包括：`bicycle, bonsai, counter, garden, kitchen, room, stump`。
*   **Tanks&Temples / Deep Blending:** 由 3D-Gaussian-Splatting 团队托管 [在此处](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)。

**自定义数据**
对于自定义数据：
1.  使用 [Colmap](https://colmap.github.io/) 处理您的图像序列，以获取 SfM 点和相机位姿。
2.  确保输出包含 `images` 文件夹和 `sparse/0` 文件夹（包含 `cameras.bin`, `images.bin`, `points3D.bin`）。
3.  将结果放入 `data/` 文件夹，遵循上述结构。

### **3.3 训练**

```
python train.py -s data/dataset_name/scenen -m output_path --appearance_dim 0
```

**注意：**
`appearance_dim` 必须设置为 0，因为它用于处理训练视图，不适合在查看器中渲染。

训练后，输出文件夹将如下所示：

```
├── cameras.json
├── cfg_args
├── input.ply
├── outputs.log
├── per_view.json
├── point_cloud
│   └── iteration_30000
│       ├── color_mlp.pt
│       ├── cov_mlp.pt
│       ├── opacity_mlp.pt
│       └── point_cloud.ply
├── results.json
├── test
│   └── ours_30000
│       ├── errors
│       ├── gt
│       ├── per_view_count.json
│       └── renders
└── train
    └── ours_30000
        ├── gt
        ├── per_view_count.json
        └── renders
```

### **3.4 导出 ONNX**

在 Scaffold-GS 环境中，使用导出脚本：

```bash
git clone -b scaffold https://github.com/Visionary-Laboratory/ONNXExample.git
cd ONNXExample

python onnx_template.py --ply output_path/point_cloud/iteration_30000/point_cloud.ply \
                        --cfg_args output_path/cfg_args \
                        --out gaussians3d_scaffold.onnx
```

---

## 4. 通用 3DGS 格式转换工具

本节介绍了如何将标准 3DGS 输出 (PLY) 转换为查看器支持的各种优化格式。

### **支持的格式与工具**

| 格式 | 扩展名 | 参考项目 | 生成 / 转换方法 |
| :--- | :--- | :--- | :--- |
| **标准** | `.ply` | [Inria 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | 直接下载或训练导出 |
| **压缩版** | `.compressed.ply` | [SuperSplat](https://github.com/playcanvas/supersplat) | [splat-transform](https://github.com/playcanvas/splat-transform) |
| **Splat** | `.splat` | [antimatter15/splat](https://github.com/antimatter15/splat) | [SuperSplat Editor](https://playcanvas.com/supersplat/editor) |
| **SPZ** | `.spz` | [nianticlabs/spz](https://github.com/nianticlabs/spz) | `ply_to_spz.py` (来自 [spz](https://github.com/nianticlabs/spz) 库) |
| **KSplat** | `.ksplat` | [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | [GaussianSplats3D 演示页面](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php) |
| **SOG** | `.sog` | [splat-transform](https://github.com/playcanvas/splat-transform) | [splat-transform](https://github.com/playcanvas/splat-transform) |


### **转换命令**

**1. 压缩版 PLY & SOG**
需要 `splat-transform`:
```bash
npm install -g @playcanvas/splat-transform
splat-transform input.ply output.compressed.ply
splat-transform input.ply output.sog
```

**2. SPZ (.spz)**
需要 Python 和 `spz` 库:
```bash
git clone https://github.com/nianticlabs/spz.git
cd spz && pip install .
python ply_to_spz.py input.ply output.spz
```

**3. KSplat (.ksplat)**
需要 Node.js:
```bash
git clone https://github.com/mkkellogg/GaussianSplats3D.git
cd GaussianSplats3D && npm install && npm run build
node util/create-ksplat.js input.ply output.ksplat
```
您也可以使用 [GaussianSplats3D 演示页面](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php) 直接导出 `.ksplat` 格式。

**4. Splat (.splat)**
使用基于 Web 的 [SuperSplat 编辑器](https://playcanvas.com/supersplat/editor) 加载 `.ply` 并导出为 `.splat`。

---

## 5. 结果可视化

一旦生成了模型（ONNX, PLY, Splat 等），您可以使用我们的查看器进行可视化。

**查看结果：**
找到生成的 ONNX 模型（通常在 Avatar 的 `./outputs/onnx` 目录中，或 4DGS/Scaffold 的 `--out` 指定路径中）并将其上传到 [Visionary 网站](https://ai4sports.opengvlab.com/index_visionary.html)。

---

## 🙏 致谢

我们衷心感谢以下项目的作者做出的精彩工作，正是这些工作使 Visionary 成为可能：

*   **Animatable Avatar:** [LHM](https://github.com/aigc3d/LHM)
*   **4DGS:** [4D-GS](https://github.com/hustvl/4DGaussians) 和 [TiNeuVox](https://github.com/hustvl/TiNeuVox)
*   **Scaffold-GS:** [Scaffold-GS](https://github.com/city-super/Scaffold-GS)
*   **Viewers & Compression:** [Inria 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [SuperSplat](https://github.com/playcanvas/supersplat), [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D).

## 📚 引用

如果您发现这些算法对您的研究有用，请考虑引用原始论文：
```bibtex
% 3D Gaussian Splatting (Original)
@article{kerbl3Dgaussians,
    author = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal = {ACM Transactions on Graphics},
    number = {4},
    volume = {42},
    month = {July},
    year = {2023},
    url = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

% Scaffold-GS
@inproceedings{scaffoldgs,
  title={Scaffold-gs: Structured 3d gaussians for view-adaptive rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20654--20664},
  year={2024}
}

% Animatable Avatar
@article{qiu2025lhm,
  title={Lhm: Large animatable human reconstruction model from a single image in seconds},
  author={Qiu, Lingteng and Gu, Xiaodong and Li, Peihao and Zuo, Qi and Shen, Weichao and Zhang, Junfei and Qiu, Kejie and Yuan, Weihao and Chen, Guanying and Dong, Zilong and others},
  journal={arXiv preprint arXiv:2503.10625},
  year={2025}
}

@inproceedings{hu2024gauhuman,
  title={Gauhuman: Articulated gaussian splatting from monocular human videos},
  author={Hu, Shoukang and Hu, Tao and Liu, Ziwei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={20418--20431},
  year={2024}
}

@article{zhan2025r3,
  title={R3-Avatar: Record and Retrieve Temporal Codebook for Reconstructing Photorealistic Human Avatars},
  author={Zhan, Yifan and Xu, Wangze and Zhu, Qingtian and Niu, Muyao and Ma, Mingze and Liu, Yifei and Zhong, Zhihang and Sun, Xiao and Zheng, Yinqiang},
  journal={arXiv preprint arXiv:2503.12751},
  year={2025}
}

% 4D Gaussian Splatting
@article{wu20234d,
  title={4d gaussian splatting for real-time dynamic scene rendering},
  author={Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
  journal={arXiv preprint arXiv:2310.08528},
  year={2023}
}

@inproceedings{TiNeuVox,
  author = {Fang, Jiemin and Yi, Taoran and Wang, Xinggang and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Nie\ss{}ner, Matthias and Tian, Qi},
  title = {Fast Dynamic Radiance Fields with Time-Aware Neural Voxels},
  year = {2022},
  booktitle = {SIGGRAPH Asia 2022 Conference Papers}
}

```