import os
import torch
# import imageio
# import numpy as np
# from PIL import Image, ImageOps
# from tqdm import tqdm
# import json
import math
# import cv2
import argparse

# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from pytorch3d.transforms import matrix_to_quaternion
# from pytorch3d.transforms.rotation_conversions import quaternion_multiply
from torchvision.transforms import v2

# from graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix_refine
from lhm_runner import HumanLRMInferrer
from LHM.models.rendering.smpl_x_voxel_dense_sampling import SMPLXVoxelMeshModel
# from diffsynth import ModelManager, WanAniCrafterCombineVideoPipeline






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--character_image_path", type=str, required=True)
    parser.add_argument("--save_gaussian_path", type=str, required=True)
    args = parser.parse_args()

    save_gaussian_path = args.save_gaussian_path
    
    os.makedirs(os.path.dirname(save_gaussian_path), exist_ok=True)

    lhm_runner = HumanLRMInferrer()

    SMPLX_MODEL = SMPLXVoxelMeshModel(
        './pretrained_models/human_model_files',
        gender="neutral",
        subdivide_num=1,
        shape_param_dim=10,
        expr_param_dim=100,
        cano_pose_type=1,
        dense_sample_points=40000,
        apply_pose_blendshape=False,
    ).cuda()

    gaussians_list, body_rgb_pil, crop_body_pil = lhm_runner.infer(
        args.character_image_path, save_gaussian_path
    )