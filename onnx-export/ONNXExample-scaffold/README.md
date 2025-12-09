# Scaffold-GS-ONNX: Exporter
A complete pipeline for running a Scaffold-GS project and exporting the trained scene representation into ONNX format for deployment.
<p align="center">
<img src="assets/pipeline.png" width=100% height=100% 
class="center">
</p>
This repository wraps around an existing [Scaffold-GS](https://github.com/city-super/Scaffold-GS) implementation, runs the reconstruction pipeline, loads the optimized Gaussians, and finally converts the model into a portable ONNX format.

---

## 📌 Features

- ✔ Help to run Scaffold-GS.
- ✔ Export ONNX format from Scaffold-GS output.

---

## 🛠️ Prerequisite
### `Configure environment`
Create a virtual environment and install the required packages 

1. Clone this repo:
```
git clone https://github.com/city-super/Scaffold-GS.git --recursive
cd Scaffold-GS
```

2. Install dependencies

```
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate scaffold_gs
```

3. Install other requirements

```
pip install onnx
```

## Data

First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organised as follows:

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


### Public Data

The BungeeNeRF dataset is available in [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA). The MipNeRF360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). And we test on scenes ```bicycle, bonsai, counter, garden, kitchen, room, stump```. The SfM data sets for Tanks&Temples and Deep Blending are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). Download and uncompress them into the ```data/``` folder.

### Custom Data

For custom data, you should process the image sequences with [Colmap](https://colmap.github.io/) to obtain the SfM points and camera poses. Then, place the results into ```data/``` folder.


## Training

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
## 📦 Exporting ONNX

In the above Scaffold-GS environment, run:

```
git clone -b scaffold https://github.com/Visionary-Laboratory/ONNXExample.git
cd ONNXExample
```
```
python onnx_template.py --ply output_path/point_cloud/iteration_30000/point_cloud.ply --cfg_args output_path/cfg_args --out gaussians3d_house.onnx
```

You will find ```gaussians4d.onnx``` in the path you specified.

## 🙏 Acknowledgments

We gratefully acknowledge the [Scaffold-GS project](https://github.com/city-super/Scaffold-GS), upon which this example is based.
```
@inproceedings{scaffoldgs,
  title={Scaffold-gs: Structured 3d gaussians for view-adaptive rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20654--20664},
  year={2024}
}
```