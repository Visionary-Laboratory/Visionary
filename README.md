# Visionary

<div align="center">

<img width="140" height="96" alt="Logo_深色竖版英文" src="https://github.com/user-attachments/assets/2d2f2c37-9fd5-438a-bb42-8163b5f8aa7a" />

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-Ready-green?style=flat-square)](https://www.w3.org/TR/webgpu/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square)](https://www.typescriptlang.org/)

[English](README.md) | [中文](README-zh.md)

**Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform**

[Project Page](https://ai4sports.opengvlab.com/) | [Document](https://your-docs-url.com) | [Quick Start](#quick-start) | [Issues](https://github.com/Visionary-Laboratory/visionary/issues)

</div>

---

> **TL;DR:** Visionary is an open, web-native platform built on WebGPU and ONNX Runtime. Enabling real-time rendering of diverse Gaussian Splatting variants (3DGS, 4DGS, Avatars) directly in the browser.

<details>
<summary><b>Abstract</b></summary>
Neural rendering, particularly 3D Gaussian Splatting (3DGS), has evolved rapidly and become a key component for building world models. In this work, we present Visionary, an open, web-native platform for real-time various Gaussian Splatting and meshes rendering. Built on an efficient WebGPU renderer with per-frame ONNX inference, Visionary enables dynamic neural processing while maintaining a lightweight, "click-to-run" browser experience. It introduces a standardized Gaussian Generator contract, which not only supports standard 3DGS rendering but also allows plug-and-play algorithms to generate or update Gaussians each frame. Such inference also enables us to apply feedforward generative post-processing. The platform further offers a plug in three.js library with a concise TypeScript API for seamless integration into existing web applications. Experiments show that, under identical 3DGS assets, Visionary achieves superior rendering efficiency compared to current Web viewers due to GPU-based primitive sorting. It already supports multiple variants, including MLP-based 3DGS, 4DGS, neural avatars, and style transformation or enhancement networks. By unifying inference and rendering directly in the browser, Visionary significantly lowers the barrier to reproduction, comparison, and deployment of 3DGS-family methods, serving as a unified World Model Carrier for both reconstructive and generative paradigms.
</details>

Additionally, we provide a powerful [Online Editor](https://ai4sports.opengvlab.com/index_visionary.html) to help users easily manage and edit 3D scenes.

![Teaser](teaser.png)

## ✨ Features

- **🚀 Native WebGPU Powered**: Utilizes `webgpu` to achieve high-performance parallel sorting and rendering of millions of Gaussian particles.
- **🎨 Hybrid Rendering Architecture**: Automatically handles depth mixing (Depth Compositing) between Gaussian point clouds and standard Meshes, perfectly solving occlusion issues and supporting complex scene compositions.
- **📦 Universal Asset Loader**: Single interface to intelligently identify and load multiple formats:
  - **Static Gaussians**: PLY, SPLAT, KSplat, SPZ, SOG
  - **Standard Mesh Models**: GLB, GLTF, FBX, OBJ
  - **4DGS/Avatar/scaffold-GS**: ONNX
  - **Your own algorithom**: See [Export your algorithm to ONNX](##Export your algorithm to ONNX) in detail.

## 🛠️ Export your algorithm to ONNX

This project supports rendering of various 3DGS/4DGS representations. To achieve this, trained 3D representations need to be exported to the ONNX format. This project provides conversion examples for 4DGS/Dynamic Avatar/Scaffold-GS, see [export instruction](/onnx-export/) for details.

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

```

## 📝 License

This project is licensed under the [Apache-2.0 license](LICENSE).
