# 🚀 Visionary: Data Preparation & ONNX Export Guide

<!-- This unified guide covers the pipeline for preparing, training, and exporting data for the **Visionary Viewer**. It includes instructions for Animatable Avatars, Dynamic Scenes (4DGS), Structured Static Scenes (Scaffold-GS), and general format conversions. -->


Visionary is built around a standardized **Gaussian Generator** contract: as long as your 3DGS-family algorithm (e.g., classic / structured / 4DGS, avatars, or any custom variant) can be exported to ONNX and outputs per-frame Gaussian attributes (position, scale, rotation, color, etc.), it can be plugged into the viewer without modifying the WebGPU renderer or shaders. The pipelines in this document (Animatable Avatar, 4DGS, Scaffold-GS, etc.) are provided as reference implementations, and you can treat them as templates when adapting your own method to the Visionary runtime.

This unified guide covers the pipeline for preparing, training, and exporting data for the **Visionary Viewer**. It includes instructions for Animatable Avatars, Dynamic Scenes (4DGS), Structured Static Scenes (Scaffold-GS), and general format conversions.


To make your own Gaussian Generator run efficiently on Visionary, we recommend a few practical ONNX export tips for the WebGPU runtime:

- **Export a graph-capture-friendly model.**  
  Try to avoid dynamic control flow and highly dynamic tensor shapes so that ONNX Runtime WebGPU can enable graph capture. A “stable” graph (fixed batch/sequence shapes, no Python/Loop-style ops, no exotic dtypes) will run significantly faster once captured and reused across frames.

- **Follow the indexing patterns used in the examples below.**  
  When you slice or index Gaussian attributes (positions, scales, rotations, colors, etc.), mirror the indexing strategy from the reference pipelines in this repo. This keeps the layout contiguous and compatible with our WebGPU kernels and post-processing utilities.

- **Replace built-in Norm ops with manual implementations.**  
  ONNX Runtime’s current WebGPU backend has known issues and performance quirks around Norm, LayerNormalization, RMSNorm. We strongly recommend exporting models where these norms have been rewritten into primitive ops (e.g., `ReduceMean` + `Sub` + `Mul` + `Add`), or using a preprocessing script to replace them before deployment.

- **Avoid huge single `Concat` / `Split` nodes.**  
  WebGPU shaders have limits on the number of resource bindings/slots. If your model uses very large `Concat` or `Split` ops over many inputs/outputs, break them into several smaller `Concat`/`Split` stages and then merge the results. This helps the WebGPU compiler stay within resource limits and improves stability.



## 📋 Table of Contents
1. [Animatable Avatar (SMPL-X based)](#1-animatable-avatar-onnx-model)
2. [4D Gaussian Splatting (Dynamic Scenes)](#2-4d-gaussian-splatting-4dgs-export)
3. [Scaffold-GS (Structured Static Scenes)](#3-scaffold-gs-export)
4. [General 3DGS Format Conversion](#4-general-3dgs-format-conversion-utilities)

---

## 1. Animatable Avatar ONNX Model

Use this pipeline to generate an animatable avatar driven by SMPL-X motion data.

### **1.1 Environment Configuration**

**Create and Activate Environment (Python 3.10)**

```bash
conda create -n visionary_avatar python==3.10 -y
conda activate visionary_avatar
```

**Install Core Frameworks**

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
pip install ninja psutil tb-nightly
```

**Install Custom Dependencies**

> **Troubleshooting Tip:** If you encounter a `ModuleNotFoundError: No module named 'torch'` error during the installation of **PyTorch3D** or **Gaussian Rasterization**, append the flag `--no-build-isolation` to the command.

```bash
# Core 3D Libraries
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/hitsz-zuoqi/sam2/
pip install git+https://github.com/XPixelGroup/BasicSR
pip install git+https://github.com/ashawkey/diff-gaussian-rasterization/

# Additional Utilities
pip install opencv-python roma smplx tqdm scikit-image huggingface_hub[cli] modelscope kornia timm accelerate diffusers==0.32.0 plyfile trimesh matplotlib jaxtyping decord transformers==4.46.2 sentencepiece chumpy gfpgan xfuser onnxruntime-gpu onnx natsort

# Downgrade Numpy for compatibility
pip install numpy==1.23.5
```

### **1.2 Model Weights & Assets**

Download pretrained models and GFPGAN weights from HuggingFace.

```bash
# Download repository
hf download MyNiuuu/Visionary_avatar --local-dir ./Visionary_avatar --local-dir-use-symlinks False

# Organize files
mv ./Visionary_avatar/pretrained_models .
mv ./Visionary_avatar/gfpgan .

# Cleanup
rm -rf ./Visionary_avatar
```

### **1.3 Motion Data Preparation (AMASS-CMU)**
Acquire the SMPL-X motion sequences from the AMASS dataset to animate the avatar.

1.  **Register/Login:** Visit [AMASS website](https://amass.is.tue.mpg.de) and log in.
2.  **Navigate to Downloads:** Go to the [Download page](https://amass.is.tue.mpg.de/download.php).
3.  **Select Dataset:** Locate the **CMU** dataset.
4.  **Download:** Click on `SMPL-X N` to download the zip file.
5.  **Extract:** Unzip the contents into the `./motions` directory.

### **1.4 Generating the ONNX Avatar Model**

**Execute the Run Script:**

```bash
bash run.sh
```
---

## 2. 4D Gaussian Splatting (4DGS) Export

Pipeline for running a 4DGS project and exporting the trained dynamic scene representation into ONNX format.

### **2.1 Environment Configuration**

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

### **2.2 Code Preparation (Crucial Step)**

To export to ONNX, you must modify `train.py` in the 4DGaussians repository to save the hex-plane AABB.

**Modify around line 299-313:**

*Change from:*
```python
tb_writer = prepare_output_and_logger(expname)
gaussians = GaussianModel(dataset.sh_degree, hyper)
dataset.model_path = args.model_path
timer = Timer()
scene = Scene(dataset, gaussians, load_coarse=None)
```

*Change to:*
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

### **2.3 Data Preparation**

*   **Synthetic Scenes:** Use [D-NeRF dataset](https://github.com/albertpumarola/D-NeRF). You can download the dataset from [dropbox](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0).

*   **Real Scenes:** Use [Neural 3D Video dataset](https://github.com/facebookresearch/Neural_3D_Video). To save the memory, please extract the frames of each video and apply [COLMAP](https://colmap.github.io/) to get initial point clouds.

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
|     ├── ...
```

### **2.4 Training**

Example training command (D-NeRF `hook` scene):

```bash
python train.py -s data/dnerf/hook --port 6017 --expname "dnerf/hook" --configs arguments/dnerf/hook.py 
```
After training, the checkpoints and output are saved in ```./output/dnerf/hook``` as follow:
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
|     ├── ...
```

### **2.5 Exporting ONNX**

In the 4D-GS environment, use the exporter script (ensure you are in the `ONNXExample` directory structure):

```bash
git clone -b 4dgs https://github.com/Visionary-Laboratory/ONNXExample.git
cd ONNXExample

python onnx_template.py --ply path/to/output/dnerf/hook/point_cloud/iteration_14000/point_cloud.ply \
                  --out your/prefered/onnxpath/gaussians4d.onnx
```
---

## 3. Scaffold-GS Export

Pipeline for running Scaffold-GS and exporting the trained static scene into ONNX.

### **3.1 Environment Configuration**

```bash
git clone https://github.com/city-super/Scaffold-GS.git --recursive
cd Scaffold-GS

# Windows only: SET DISTUTILS_USE_SDK=1 
conda env create --file environment.yml
conda activate scaffold_gs
pip install onnx
```

### **3.2 Data Preparation**

Create a `data/` folder. Data should follow the standard Colmap structure:

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

**Public Data**
You can download standard datasets and uncompress them into the `data/` folder:

*   **BungeeNeRF:** Available on [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing) or [Baidu Netdisk (Code: 4whv)](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA).
*   **MipNeRF360:** Provided by the paper author [here](https://jonbarron.info/mipnerf360/). We test on scenes: `bicycle, bonsai, counter, garden, kitchen, room, stump`.
*   **Tanks&Temples / Deep Blending:** Hosted by the 3D-Gaussian-Splatting team [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

**Custom Data**
For custom data:
1.  Process your image sequences with [Colmap](https://colmap.github.io/) to obtain the SfM points and camera poses.
2.  Ensure the output contains the `images` folder and the `sparse/0` folder (containing `cameras.bin`, `images.bin`, `points3D.bin`).
3.  Place the results into the `data/` folder following the structure above.

### **3.3 Training**

```
python train.py -s data/dataset_name/scenen -m output_path --appearance_dim 0
```

**Note:**  
`appearance_dim` must be set to 0, as it is used for processing the training view and is not suitable for rendering in the viewer.

After Training, the output folder will be like:

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

### **3.4 Exporting ONNX**

In the Scaffold-GS environment, use the exporter script:

```bash
git clone -b scaffold https://github.com/Visionary-Laboratory/ONNXExample.git
cd ONNXExample

python onnx_template.py --ply output_path/point_cloud/iteration_30000/point_cloud.ply \
                        --cfg_args output_path/cfg_args \
                        --out gaussians3d_scaffold.onnx
```

---

## 4. General 3DGS Format Conversion Utilities

This section describes how to convert standard 3DGS outputs (PLY) into various optimized formats supported by the viewer.

### **Supported Formats & Tools**

| Format | Extension | Reference Project | Generation / Conversion Method |
| :--- | :--- | :--- | :--- |
| **Standard** | `.ply` | [Inria 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | Direct Download or Training Export|
| **Compressed** | `.compressed.ply` | [SuperSplat](https://github.com/playcanvas/supersplat) | [splat-transform](https://github.com/playcanvas/splat-transform) |
| **Splat** | `.splat` | [antimatter15/splat](https://github.com/antimatter15/splat) | [SuperSplat Editor](https://playcanvas.com/supersplat/editor) |
| **SPZ** | `.spz` | [nianticlabs/spz](https://github.com/nianticlabs/spz) | `ply_to_spz.py` (from [spz](https://github.com/nianticlabs/spz) lib) |
| **KSplat** | `.ksplat` | [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | [GaussianSplats3D Demo Page](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php) |
| **SOG** | `.sog` | [splat-transform](https://github.com/playcanvas/splat-transform) | [splat-transform](https://github.com/playcanvas/splat-transform) |


### **Conversion Commands**

**1. Compressed PLY & SOG**
Requires `splat-transform`:
```bash
npm install -g @playcanvas/splat-transform
splat-transform input.ply output.compressed.ply
splat-transform input.ply output.sog
```

**2. SPZ (.spz)**
Requires Python and `spz` lib:
```bash
git clone https://github.com/nianticlabs/spz.git
cd spz && pip install .
python ply_to_spz.py input.ply output.spz
```

**3. KSplat (.ksplat)**
Requires Node.js:
```bash
git clone https://github.com/mkkellogg/GaussianSplats3D.git
cd GaussianSplats3D && npm install && npm run build
node util/create-ksplat.js input.ply output.ksplat
```
You can also use [GaussianSplats3D Demo Page](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php) to export `.ksplat` format directly.

**4. Splat (.splat)**
Use the web-based [SuperSplat Editor](https://playcanvas.com/supersplat/editor) to load a `.ply` and export as `.splat`.

---

## 5. Visualizing the Results

Once you have generated your model (ONNX, PLY, Splat, etc.), you can visualize it using our viewer.

**View Results:**
Locate the generated ONNX model (typically in the `./outputs/onnx` directory for Avatar, or the path you specified in `--out` for 4DGS/Scaffold) and upload it to the [Visionary Website](https://ai4sports.opengvlab.com/index_visionary.html).

---

## 🙏 Acknowledgments

We sincerely thank the authors of the following projects for their wonderful work, which makes Visionary possible:

*   **Animatable Avatar:** [LHM](https://github.com/aigc3d/LHM)
*   **4DGS:** [4D-GS](https://github.com/hustvl/4DGaussians) and [TiNeuVox](https://github.com/hustvl/TiNeuVox)
*   **Scaffold-GS:** [Scaffold-GS](https://github.com/city-super/Scaffold-GS)
*   **Viewers & Compression:** [Inria 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [SuperSplat](https://github.com/playcanvas/supersplat), [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D).

## 📚 Citations

If you find these algorithms useful for your research, please consider citing the original papers:

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