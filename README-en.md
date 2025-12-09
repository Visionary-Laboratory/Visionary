# VisionaryCore

<div align="center">

![VisionaryCore Logo](public/vite.svg)

[![NPM Version](https://img.shields.io/npm/v/visionary-core?style=flat-square)](https://www.npmjs.com/package/visionary-core)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-Ready-green?style=flat-square)](https://www.w3.org/TR/webgpu/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square)](https://www.typescriptlang.org/)

[English](README-en.md) | [中文](README.md)

**Next-Generation WebGPU-Powered 3D Gaussian Splatting Rendering Engine**

[VisionaryEditor](https://editor-url-placeholder) | [Online Docs](https://your-docs-url.com) | [Quick Start](#quick-start) | [FAQ](#faq)

</div>

---

**VisionaryCore** is a high-performance immersive Web 3D rendering engine built on **WebGPU** and **onnx-runtime** technologies.

It aims to break the boundaries between traditional 3D meshes and emerging point cloud rendering technologies. Through an original **Hybrid Rendering Pipeline**, it seamlessly blends standard 3D models (GLB/GLTF) with high-fidelity Gaussian Splatting (3DGS/4DGS) content within the same WebGPU context, providing unified depth occlusion and lighting processing.

Additionally, we provide a powerful [Online Editor](https://editor-url-placeholder) to help users easily manage and edit 3D scenes.

## ✨ Core Features

- **🚀 Native WebGPU Powered**: Utilizes `three/webgpu` and a custom Compute Shader rasterizer to achieve high-performance parallel sorting and rendering of millions of Gaussian particles.
- **🎨 Hybrid Rendering Architecture**: Automatically handles depth mixing (Depth Compositing) between Gaussian point clouds and standard Meshes, perfectly solving occlusion issues and supporting complex scene compositions.
- **📦 Universal Asset Loader**: Single interface to intelligently identify and load multiple formats:
  - **Static Gaussians**: PLY, SPLAT, KSplat, SPZ, SOG
  - **Standard Models**: GLB, GLTF, FBX, OBJ
  - **Dynamic Gaussians**: ONNX (4DGS)
- **🧠 Powerful AI Inference**: Deeply integrated with **ONNX Runtime Web (ORT)** to support real-time decoding and playback of 4D dynamic Gaussian models, delivering cinematic dynamic visual experiences.
- **🛠️ Developer Friendly**: Provides a modular API based on TypeScript, easy to integrate into existing Web applications.

## 🚀 Quick Start

### 1. Install Dependencies

Ensure that [Node.js](https://nodejs.org/) (v18+ recommended) is installed in your environment.

```bash
# Clone the repository
git clone https://github.com/your-username/Visionary-PrePublic.git
cd Visionary-PrePublic

# Install dependencies
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

After successful startup, visit the following address to view the example:
👉 **http://localhost:8901/demo/index.html**

### 3. Model Assets

You can import our provided [example assets](https://editor-url-placeholder) or your own 3DGS/4DGS assets in the page. For details on creating 4DGS assets, see [Convert to ONNX](#convert-to-onnx).

## 🧠 Convert to ONNX

This project supports rendering of various 3DGS/4DGS representations. To achieve this, trained 3D representations need to be exported to the ONNX format. This project provides conversion examples for 4DGS/Dynamic Avatar/Scaffold-GS, see [/examples](/examples/README.md) for details.

Taking 4DGS conversion as an example:

### **Environment Configuration**

```bash
git clone https://github.com/hustvl/4DGaussians
cd 4DGaussians
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7 
conda activate Gaussians4D
    
# Install requirements
pip install -r requirements.txt
pip install onnx

# Install submodules
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

### **Code Preparation**

To export to ONNX, you must modify `train.py` in the 4DGaussians repository to save the hex-plane AABB.

**Modify around line 299-313:**

*Before Change:*
```python
tb_writer = prepare_output_and_logger(expname)
gaussians = GaussianModel(dataset.sh_degree, hyper)
dataset.model_path = args.model_path
timer = Timer()
scene = Scene(dataset, gaussians, load_coarse=None)
```

*After Change:*
```python
args.model_path = os.path.join("./output/", expname)
os.makedirs(args.model_path, exist_ok = True)
gaussians = GaussianModel(dataset.sh_degree, hyper)
dataset.model_path = args.model_path
timer = Timer()
scene = Scene(dataset, gaussians, load_coarse=None)
# ADDED FOR ONNX EXPORT:
grid_aabb = scene.gaussians._deformation.deformation_net.get_aabb
args.grid_aabb = [x.cpu().tolist() for x in grid_aabb]
tb_writer = prepare_output_and_logger(expname)
```

*To ensure consistent paths, **remove** the subsequent automatic `args.model_path` generation logic (UUID generation):*
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

### **Data Preparation**

*   **Synthetic Scenes:** Use the [D-NeRF dataset](https://github.com/albertpumarola/D-NeRF). You can download the dataset from [dropbox](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0).

*   **Real Scenes:** Use the [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video). To save memory, please extract frames from each video and apply [COLMAP](https://colmap.github.io/) to get initial point clouds.

    1.  **Extract Frames:**
        ```bash
        python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
        ```

    2.  **Generate Point Clouds:**
        ```bash
        bash colmap.sh data/dynerf/cut_roasted_beef llff
        ```

    3.  **Downsample Point Clouds:**
        ```bash
        python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
        ```

**Directory Structure**
The final datasets should be organized as follows:

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

### **Training**

Example training command (D-NeRF `hook` scene):

```bash
python train.py -s data/dnerf/hook --port 6017 --expname "dnerf/hook" --configs arguments/dnerf/hook.py 
```
After training, the checkpoints and output are saved in `./output/dnerf/hook` as follows:
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

### **Export to ONNX**

In the 4D-GS environment, use the exporter script (ensure you are in the `ONNXExample` directory structure):

```bash
git clone -b 4dgs https://github.com/Visionary-Laboratory/ONNXExample.git
cd ONNXExample

python onnx_template.py --ply path/to/output/dnerf/hook/point_cloud/iteration_14000/point_cloud.ply \
                  --out your/prefered/onnxpath/gaussians4d.onnx
```

## 🤝 Contributions & Acknowledgments

This project is deeply inspired and supported by the following open source projects:

- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)**
- **[Three.js](https://threejs.org/)**
- **[ONNX Runtime Web](https://onnxruntime.ai/)**
- **[web-splat](https://github.com/KeKsBoTer/web-splat/)**
- **[image-to-line-drawing](https://github.com/luckycucu/image-to-line-drawing/)**

## 📄 Citation

If you use VisionaryCore in your research or projects, please consider citing:

```bibtex
@misc{visionarycore2025,
  author = {Your Name and Contributors},
  title = {VisionaryCore: High-Performance WebGPU 3D Gaussian Splatting Renderer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/Visionary-PrePublic}}
}
```

## 📝 License

This project is licensed under the [MIT License](LICENSE).

