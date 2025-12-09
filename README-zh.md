# Visionary

<div align="center">

<img width="140" height="96" alt="Logo_深色竖版英文" src="https://github.com/user-attachments/assets/2d2f2c37-9fd5-438a-bb42-8163b5f8aa7a" />

[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![WebGPU](https://img.shields.io/badge/WebGPU-Ready-green?style=flat-square)](https://www.w3.org/TR/webgpu/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square)](https://www.typescriptlang.org/)

[English](README.md) | [中文](README-zh.md)

**Visionary: 基于 WebGPU 高斯泼溅平台构建的世界模型载体**

[VisionaryEditor](https://editor-url-placeholder) | [在线文档](https://your-docs-url.com) | [快速开始](#快速开始) | [Issues](https://github.com/Visionary-Laboratory/visionary/issues)

</div>

---

> **TL;DR:** Visionary 是一个基于 WebGPU 和 ONNX Runtime 构建的开放、原生 Web 平台。它支持直接在浏览器中实时渲染多种 Gaussian Splatting 变体（3DGS、4DGS、Avatars）。

<details>
<summary><b>Abstract</b></summary>
神经渲染，特别是 3D Gaussian Splatting (3DGS)，发展迅速并已成为构建世界模型的关键组件。在这项工作中，我们推出了 Visionary，一个开放的原生 Web 平台，用于实时渲染各种 Gaussian Splatting 和网格。Visionary 基于高效的 WebGPU 渲染器和逐帧 ONNX 推理构建，能够在保持轻量级、"点击即运行"的浏览器体验的同时，实现动态神经处理。它引入了标准化的 Gaussian Generator 契约，不仅支持标准 3DGS 渲染，还允许即插即用的算法在每帧生成或更新高斯体。这种推理能力也使我们能够应用前馈生成式后处理。该平台进一步提供了一个插件式的 three.js 库，并具有简洁的 TypeScript API，可无缝集成到现有的 Web 应用程序中。实验表明，在相同的 3DGS 资产下，由于基于 GPU 的图元排序，Visionary 实现了优于当前 Web 查看器的渲染效率。它已经支持多种变体，包括基于 MLP 的 3DGS、4DGS、神经 Avatar 以及风格迁移或增强网络。通过直接在浏览器中统一推理和渲染，Visionary 显著降低了 3DGS 系列方法的复现、比较和部署门槛，作为一个统一的世界模型载体，服务于重建和生成范式。
</details>

此外，我们还提供了一个功能强大的 [在线编辑器](https://editor-url-placeholder)，帮助用户轻松管理和编辑 3D 场景。

![Teaser](teaser.png)

## ✨ 项目特色

- **🚀 原生 WebGPU 驱动**：利用 `three/webgpu` 与自定义 Compute Shader 光栅化器，实现百万级高斯粒子的高性能并行排序与渲染。
- **🎨 混合渲染架构**：自动处理高斯点云与标准 Mesh 的深度混合（Depth Compositing），完美解决遮挡问题，支持复杂的场景组合。
- **📦 统一资源加载 (Universal Loader)**：单一接口智能识别并加载多种格式：
  - **静态高斯**: PLY, SPLAT, KSplat, SPZ, SOG
  - **标准模型**: GLB, GLTF, FBX, OBJs
  - **4DGS/Avatar/scaffold-GS**: ONNX
  - **自定义算法**: 详见 [导出算法至 ONNX](#导出算法至-onnx)。

<a id="快速开始"></a>
## 🚀 快速开始

### 1. 安装依赖

确保您的环境中已安装 [Node.js](https://nodejs.org/) (推荐 v18+)。

```bash
# 克隆仓库
git clone https://github.com/Visionary-Laboratory/visionary.git
cd visionary

# 安装依赖
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

启动成功后，访问以下地址查看示例：
👉 **http://localhost:3000/demo/index.html**

### 3. 模型资产

可以在页面中导入我们提供的[示例资产](https://ai4sports.opengvlab.com/models/trex.onnx)，或自己的3DGS/4DGS资产。4DGS资产制作详见[转化ONNX格式](#转化ONNX格式)。

<a id="导出算法至-onnx"></a>
## 🛠️ 导出算法至 ONNX

本项目支持多种 3DGS/4DGS 表示的渲染。要做到这一点，需要将训练好的 3D 表示导出为 ONNX 格式。本项目提供了 4DGS/Dynamic Avatar/Scaffold-GS 的转换示例，详见[导出说明](/onnx-export/)。

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
