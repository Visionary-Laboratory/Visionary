## **Animatable Avatar ONNX Model within the Visionary Project**


### **1. Environment Configuration**

To begin, you must create a specific Conda environment and install the required dependencies. This setup utilizes Python 3.10.

**Create and Activate Environment**

```bash
conda create -n visionary_avatar python==3.10 -y
conda activate visionary_avatar
```

**Install Core Frameworks**

Install PyTorch, torchvision, and xformers with CUDA support.

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
pip install ninja psutil tb-nightly
```

**Install Custom Dependencies**

The following packages are installed directly from their repositories.

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

-----

### **2. Model Weights & Assets**

You need to download the pretrained models and the GFPGAN weights. The commands below will fetch them from HuggingFace and organize them into the root directory.

```bash
# Download repository
hf download MyNiuuu/Visionary_avatar --local-dir ./Visionary_avatar --local-dir-use-symlinks False

# Organize files
mv ./Visionary_avatar/pretrained_models .
mv ./Visionary_avatar/gfpgan .

# Cleanup
rm -rf ./Visionary_avatar
```

-----

### **3. Motion Data Preparation (AMASS-CMU)**

To animate the avatar, you must acquire the SMPL-X motion sequences from the AMASS dataset.

1.  **Register/Login:** Visit the [AMASS website](https://amass.is.tue.mpg.de) and log in.
2.  **Navigate to Downloads:** Go to the [Download page](https://amass.is.tue.mpg.de/download.php).
3.  **Select Dataset:** Locate the **CMU** dataset.
    <td align="center">
    <img src="assets/AMASS_CMU.png"/>
    </td>
4.  **Download:** Click on `SMPL-X N` to download the zip file.
5.  **Extract:** Unzip the contents into the `./motions` directory.

-----

### **4. Generating the ONNX Avatar Model**

Once the environment and data are ready, you can generate the ONNX avatar model.

**Execute the Run Script:**

```bash
bash run.sh
```

**View Results:**

1.  Locate the generated ONNX model in the `./outputs/onnx` directory.
2.  Visit the [Visionary Website](https://ai4sports.opengvlab.com/index_visionary.html).
3.  Upload your ONNX file to visualize the result.



### **5. Acknowledgement**

We use [LHM](https://github.com/aigc3d/LHM) to generate our animatable avatar. We sincerely thank the LHM team for their wonderful work.