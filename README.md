# Visionary

<div align="center">

<img width="140" height="96" alt="Logo_深色竖版英文" src="https://github.com/user-attachments/assets/2d2f2c37-9fd5-438a-bb42-8163b5f8aa7a" />

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-Ready-green?style=flat-square)](https://www.w3.org/TR/webgpu/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square)](https://www.typescriptlang.org/)

[English](README.md) | [中文](README-zh.md)

**Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform**

[Project Page](https://ai4sports.opengvlab.com/) | [Paper](https://arxiv.org/) | [Document](https://your-docs-url.com)

</div>

---

> **TL;DR:** Visionary is an open, web-native platform built on WebGPU and ONNX Runtime. Enabling real-time rendering of diverse Gaussian Splatting variants (3DGS, MLP-based 3DGS, 4DGS, Neural Avatars and <span style="font-family: 'Brush Script MT', cursive; font-size: 1.2em; color: #FFD700; text-shadow: 1px 1px 2px black;">✨any future algorithms✨</span>), and traditional 3d Mesh, directly in the browser. We also support post-processing using feed-forward networks.

<details>
<summary><b>Abstract</b></summary>
Neural rendering, particularly 3D Gaussian Splatting (3DGS), has evolved rapidly and become a key component for building world models. In this work, we present Visionary, an open, web-native platform for real-time various Gaussian Splatting and meshes rendering. Built on an efficient WebGPU renderer with per-frame ONNX inference, Visionary enables dynamic neural processing while maintaining a lightweight, "click-to-run" browser experience. It introduces a standardized Gaussian Generator contract, which not only supports standard 3DGS rendering but also allows plug-and-play algorithms to generate or update Gaussians each frame. Such inference also enables us to apply feedforward generative post-processing. The platform further offers a plug in three.js library with a concise TypeScript API for seamless integration into existing web applications. Experiments show that, under identical 3DGS assets, Visionary achieves superior rendering efficiency compared to current Web viewers due to GPU-based primitive sorting. It already supports multiple variants, including MLP-based 3DGS, 4DGS, neural avatars, and style transformation or enhancement networks. By unifying inference and rendering directly in the browser, Visionary significantly lowers the barrier to reproduction, comparison, and deployment of 3DGS-family methods, serving as a unified World Model Carrier for both reconstructive and generative paradigms.
</details>

We have developed a powerful [Online Editor](https://ai4sports.opengvlab.com/index_visionary.html) based on Visionary to help users easily manage and edit 3D scenes in one click. If you want to develop your own web project using this project, please refer to [Quick Start](#quick-start).

![Teaser](teaser.png)

## ✨ Features

- **🚀 Native WebGPU Powered**: Utilizes `webgpu` to achieve high-performance parallel sorting and rendering of millions of Gaussian particles.
- **🎨 Hybrid Rendering Architecture**: Automatically handles depth mixing (Depth Compositing) between Gaussian point clouds and standard Meshes, perfectly solving occlusion issues and supporting complex scene compositions.
- **📦 Universal Asset Loader**: Single interface to intelligently identify and load multiple formats:
  - **Static Gaussians**: PLY, SPLAT, KSplat, SPZ, SOG
  - **Standard Mesh Models**: GLB, GLTF, FBX, OBJ
  - **4DGS/Avatar/scaffold-GS**: ONNX
  - **<span style="font-family: 'Brush Script MT', cursive; font-size: 1.1em; color: #FF4500;">🔥Your own algorithom</span>**: See [Export your algorithm to ONNX](onnx-export/README.md) in detail.

<a id="quick-start"></a>
## 🚀 Quick Start

### 1. Install Dependencies

Ensure that [Node.js](https://nodejs.org/) (v18+ recommended) is installed in your environment.

```bash
# Clone the repository
git clone https://github.com/Visionary-Laboratory/visionary.git
cd visionary

# Install dependencies
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

After successful startup, visit the following address to view the example:
👉 **http://localhost:3000/demo/index.html**

### 3. Model Assets

You can import our provided [example assets](https://ai4sports.opengvlab.com/models/trex.onnx) or your own 3DGS/4DGS assets in the page. For details on creating 4DGS assets, see [Convert to ONNX](#convert-to-onnx).

<a id="export-your-algorithm-to-onnx"></a>
## 🛠️ Export your algorithm to ONNX

This project supports rendering of various 3DGS/4DGS representations. To achieve this, trained 3D representations need to be exported to the ONNX format. This project provides conversion examples for 4DGS/Dynamic Avatar/Scaffold-GS, see [export instruction](onnx-export/README.md) for details.

## 🤝 Contributions & Acknowledgments

This project is deeply inspired and supported by the following open source projects:

- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)**
- **[Three.js](https://threejs.org/)**
- **[ONNX Runtime Web](https://onnxruntime.ai/)**
- **[web-splat](https://github.com/KeKsBoTer/web-splat/)**
- **[image-to-line-drawing](https://github.com/luckycucu/image-to-line-drawing/)**

## 📄 Citation

If you use Visionary in your research or projects, please consider citing:

```bibtex
@article{gong2025visionary,
      title={Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform}, 
      author={Gong, Yuning and Liu, Yifei and Zhan, Yifan and Niu, Muyao and Li, Xueying and Liao, Yuanjun and Chen, Jiaming and Gao, Yuanyuan and Chen, Jiaqi and Chen, Minming and Zhou, Li and Zhang, Yuning and Wang, Wei and Hou, Xiaoqing and Huang, Huaxi and Tang, Shixiang and Ma, Le and Zhang, Dingwen and Yang, Xue and Yan, Junchi and Zhang, Yanchi and Zheng, Yinqiang and Sun, Xiao and Zhong, Zhihang},
      journal={arXiv preprint arXiv:xxxx.xxxxx},
      year={2025}
}
```

## 📝 License

This project is licensed under the [Apache-2.0 license](LICENSE).
