# Visionary

<div align="center">

<img width="140" height="96" alt="Logo_深色竖版英文" src="https://github.com/user-attachments/assets/2d2f2c37-9fd5-438a-bb42-8163b5f8aa7a" />

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-Ready-green?style=flat-square)](https://www.w3.org/TR/webgpu/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square)](https://www.typescriptlang.org/)

[English](README.md) | [中文](README-zh.md)

**Visionary: 基于 WebGPU 高斯泼溅平台构建的世界模型载体**

[VisionaryEditor](https://editor-url-placeholder) | [在线文档](https://your-docs-url.com) | [快速开始](#快速开始) | [常见问题](#faq)

</div>

---

**Visionary** 是一个基于 **WebGPU** 和 **onnx-runtime** 技术构建的高性能沉浸式 Web 3D 渲染引擎。

它旨在打破传统 3D 网格与新兴点云渲染技术的界限，通过独创的 **混合渲染管线 (Hybrid Rendering Pipeline)**，实现在同一 WebGPU 上下文中无缝融合标准 3D 模型（GLB/GLTF）与高保真 Gaussian Splatting（3DGS/4DGS）内容，并提供统一的深度遮挡与光照处理。

此外，我们还提供了一个功能强大的 [在线编辑器](https://editor-url-placeholder)，帮助用户轻松管理和编辑 3D 场景。

![Teaser](teaser.pdf)

## ✨ 核心特性

- **🚀 原生 WebGPU 驱动**：利用 `three/webgpu` 与自定义 Compute Shader 光栅化器，实现百万级高斯粒子的高性能并行排序与渲染。
- **🎨 混合渲染架构**：自动处理高斯点云与标准 Mesh 的深度混合（Depth Compositing），完美解决遮挡问题，支持复杂的场景组合。
- **📦 统一资源加载 (Universal Loader)**：单一接口智能识别并加载多种格式：
  - **静态高斯**: PLY, SPLAT, KSplat, SPZ, SOG
  - **标准模型**: GLB, GLTF, FBX, OBJ
  - **动态高斯**: ONNX (4DGS)
- **🧠 强大的 AI 推理**：深度集成 **ONNX Runtime Web (ORT)**，支持 4D 动态高斯模型的实时解码与播放，带来电影级的动态视觉体验。
- **🛠️ 开发者友好**：提供基于 TypeScript 的模块化 API，易于集成到现有的 Web 应用中。

## 🚀 快速开始

### 1. 安装依赖

确保您的环境中已安装 [Node.js](https://nodejs.org/) (推荐 v18+)。

```bash
# 克隆仓库
git clone https://github.com/your-username/Visionary-PrePublic.git
cd Visionary-PrePublic

# 安装依赖
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

启动成功后，访问以下地址查看示例：
👉 **http://localhost:8901/demo/index.html**

### 3. 模型资产

可以在页面中导入我们提供的[示例资产](https://editor-url-placeholder)，或自己的3DGS/4DGS资产。4DGS资产制作详见[转化ONNX格式](#转化ONNX格式)。

## 🧠 转化ONNX格式

本项目支持多种3DGS/4DGS表示的渲染，要做到这一点，需要将训练好的3D表示导出为 ONNX 格式。本项目提供了4DGS/动态Avatar/scaffold-GS的转化示例，详见[/examples](examples/README-ZH.md)。

## 🤝 贡献与致谢

本项目深受以下开源项目的启发与支持，特此致谢：

- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)**
- **[Three.js](https://threejs.org/)**
- **[ONNX Runtime Web](https://onnxruntime.ai/)**
- **[web-splat](https://github.com/KeKsBoTer/web-splat/)**
- **[image-to-line-drawing](https://github.com/luckycucu/image-to-line-drawing/)**

## 📄 引用 (Citation)

如果您在研究或项目中使用了 Visionary，请考虑引用：

```bibtex

```

## 📝 License

本项目采用 [Apache-2.0 License](LICENSE) 许可证。
