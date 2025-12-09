# 📂 3DGS Data Preparation Guide

This viewer supports various 3D Gaussian Splatting formats. Below is a quick reference table and detailed conversion instructions for each format.

## 📊 Supported Formats

| Format | Extension | Reference Project | Generation / Conversion Method |
| :--- | :--- | :--- | :--- |
| **Standard** | `.ply` | [Inria 3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | Direct Download or Training Export|
| **Compressed** | `.compressed.ply` | [SuperSplat](https://github.com/playcanvas/supersplat) | [splat-transform](https://github.com/playcanvas/splat-transform) |
| **Splat** | `.splat` | [antimatter15/splat](https://github.com/antimatter15/splat) | [SuperSplat Editor](https://playcanvas.com/supersplat/editor) |
| **SPZ** | `.spz` | [nianticlabs/spz](https://github.com/nianticlabs/spz) | `ply_to_spz.py` (uses [spz](https://github.com/nianticlabs/spz) lib) |
| **KSplat** | `.ksplat` | [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | [GaussianSplats3D Demo Page](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php) |
| **SOG** | `.sog` | [splat-transform](https://github.com/playcanvas/splat-transform) | [splat-transform](https://github.com/playcanvas/splat-transform) |


---

### 1. PLY (`.ply`)

The viewer supports standard PLY files and automatically distinguishes between **3D Gaussian Splatting** point clouds and standard **3D Meshes**.

*   **For 3DGS:** Use the output from your training (e.g., standard Inria implementation) or download [official datasets](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/).
*   **For Meshes:** Use standard exports from Blender, MeshLab, or other 3D modeling software.

### 2. Compressed PLY (`.compressed.ply`)

Convert existing PLY files using the `splat-transform` tool to reduce file size.

**Installation:**
```bash
npm install -g @playcanvas/splat-transform
```

**Usage:**
```bash
splat-transform input.ply output.compressed.ply
```

### 3. SPZ (`.spz`)

Use the provided Python script to convert PLY files to the highly compressed SPZ format.

**Prerequisites:**
```bash
git clone https://github.com/nianticlabs/spz.git
cd spz
pip install .
```

**Usage:**
```bash
python ply_to_spz.py input.ply output.spz
```

### 4. Splat (`.splat`)

For `.splat` files, the easiest method is to use the web-based editor:

1.  Go to the [SuperSplat Editor](https://playcanvas.com/supersplat/editor).
2.  Load your `.ply` file.
3.  Export as `.splat`.

### 5. KSplat (`.ksplat`)

You can generate `.ksplat` files from [GaussianSplats3D Demo Page](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php) or using the official repository tools:

**Setup:**
```bash
git clone https://github.com/mkkellogg/GaussianSplats3D.git
cd GaussianSplats3D
npm install
npm run build # (Or 'npm run build-windows' on Windows)
```

**Usage:**
```bash
node util/create-ksplat.js input.ply output.ksplat
```

### 6. SOG (`.sog`)

Similar to Compressed PLY, use the `splat-transform` CLI tool.

**Installation:**
```bash
npm install -g @playcanvas/splat-transform
```

**Usage:**
```bash
splat-transform input.ply output.sog
```