# 🚀 4DGS-ONNX Exporter  
A complete pipeline for running a 4D Gaussian Splatting (4DGS) project and exporting the trained dynamic scene representation into ONNX format for deployment.

![image](https://github.com/Visionary-Laboratory/ONNXExample/blob/4dgs/images/teaser.png)

This repository wraps around an existing 4DGS implementation ([4D-GS](https://github.com/hustvl/4DGaussians?tab=readme-ov-file)), runs the reconstruction pipeline, loads the optimized Gaussians, and finally converts the model into a portable ONNX format.

---

## 📌 Features

- ✔ Help to run 4D-GS.
- ✔ Export ONNX format from 4D-GS output.

---

## 🛠️ Prerequisite
### `Configure environment`
Create a virtual environment and install the required packages 

    git clone https://github.com/hustvl/4DGaussians
    cd 4DGaussians
    git submodule update --init --recursive
    conda create -n Gaussians4D python=3.7 
    conda activate Gaussians4D
    
Install other requirements:

    pip install -r requirements.txt
    pip install onnx

Install submodules:

    pip install -e submodules/depth-diff-gaussian-rasterization
    pip install -e submodules/simple-knn


### `Data Preparation`

For synthetic scenes: The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is used. You can download the dataset from [dropbox](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0).

For Neural 3D Data, please refer to this [link](https://github.com/facebookresearch/Neural_3D_Video). To save the memory, please extract the frames of each video. Then apply [COLMAP](https://colmap.github.io/) to get initial point clouds.

    # First, extract the frames of each video.
    python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
    
    # Second, generate point clouds from input data.
    bash colmap.sh data/dynerf/cut_roasted_beef llff
    
    # Third, downsample the point clouds generated in the second step.
    python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply

The final datasets are organized as follow:

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

### `Code Preparation`

Please save the ```aabb``` of hex-plane with modifications around ```train.py``` [line 299](https://github.com/hustvl/4DGaussians/blob/843d5ac636c37e4b611242287754f3d4ed150144/train.py#L299) and [line 313](https://github.com/hustvl/4DGaussians/blob/843d5ac636c37e4b611242287754f3d4ed150144/train.py#L313) in [4D-GS](https://github.com/hustvl/4DGaussians?tab=readme-ov-file).
From:

    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)

to:

    args.model_path = os.path.join("./output/", expname)
    os.makedirs(args.model_path, exist_ok = True)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    grid_aabb = scene.gaussians._deformation.deformation_net.get_aabb
    args.grid_aabb = [x.cpu().tolist() for x in grid_aabb]
    tb_writer = prepare_output_and_logger(expname)

and remove:

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

The final code looks like:

![image](https://github.com/Visionary-Laboratory/ONNXExample/blob/4dgs/images/code_train.PNG)

---


## 🧠 Training

To train synthetic and real (dnerf ```hook``` as example), run:

    python train.py -s data/dnerf/hook --port 6017 --expname "dnerf/hook" --configs arguments/dnerf/hook.py 


After training, the checkpoints and output are saved in ```./output/dnerf/hook``` as follow:

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

---

## 📦 Exporting ONNX

In the above 4D-GS environment, run:

    git clone -b 4dgs https://github.com/Visionary-Laboratory/ONNXExample.git
    cd ONNXExample
    
    python onnx_template.py --ply path/to/output/dnerf/hook/point_cloud/iteration_14000/point_cloud.ply \
                      --out your/prefered/onnxpath/gaussians4d.onnx

You will find ```gaussians4d.onnx``` in the path you specified.

---

## 🙏 Acknowledgments

We gratefully acknowledge the [4D-GS project](https://github.com/hustvl/4DGaussians), upon which this example is based.

Please also cite [TiNeuVox](https://github.com/hustvl/TiNeuVox) for insights about neural voxel grids and dynamic scenes reconstruction.

    @InProceedings{Wu_2024_CVPR,
        author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
        title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2024},
        pages     = {20310-20320}
    }
    
    @inproceedings{TiNeuVox,
      author = {Fang, Jiemin and Yi, Taoran and Wang, Xinggang and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Nie\ss{}ner, Matthias and Tian, Qi},
      title = {Fast Dynamic Radiance Fields with Time-Aware Neural Voxels},
      year = {2022},
      booktitle = {SIGGRAPH Asia 2022 Conference Papers}
    }




