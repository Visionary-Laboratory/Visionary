# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu  & Xiaodong Gu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:30:37
# @Function      : Inference code for human_lrm model

import argparse
import os
import pdb
import time

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from engine.pose_estimation.pose_estimator import PoseEstimator
from engine.SegmentAPI.base import Bbox
from LHM.utils.model_download_utils import AutoModelQuery

try:
    from engine.SegmentAPI.SAM import SAM2Seg
except:
    print("\033[31mNo SAM2 found! Try using rembg to remove the background. This may slightly degrade the quality of the results!\033[0m")
    from rembg import remove

from LHM.datasets.cam_utils import (
    build_camera_principle,
    build_camera_standard,
    create_intrinsics,
    surrounding_views_linspace,
)
from LHM.models.modeling_human_lrm import ModelHumanLRM
from LHM.runners import REGISTRY_RUNNERS
from LHM.runners.infer.utils import (
    calc_new_tgt_size_by_aspect,
    center_crop_according_to_mask,
    prepare_motion_seqs,
    resize_image_keepaspect_np,
)
from LHM.utils.download_utils import download_extract_tar_from_url
from LHM.utils.face_detector import FaceDetector

# from LHM.utils.video import images_to_video
from LHM.utils.ffmpeg_utils import images_to_video
from LHM.utils.hf_hub import wrap_model_hub
from LHM.utils.logging import configure_logger
from LHM.utils.model_card import MODEL_CARD, MODEL_CONFIG


def prior_check():
    if not os.path.exists('./pretrained_models'):
        prior_data = MODEL_CARD['prior_model']
        download_extract_tar_from_url(prior_data)


from .base_inferrer import Inferrer

logger = get_logger(__name__)


def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device

def resize_with_padding(img, target_size, padding_color=(255, 255, 255)):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    dw = target_w - new_w
    dh = target_h - new_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    return padded


def get_bbox(mask):
    height, width = mask.shape
    pha = mask / 255.0
    pha[pha < 0.5] = 0.0
    pha[pha >= 0.5] = 1.0

    # obtain bbox
    _h, _w = np.where(pha == 1)

    whwh = [
        _w.min().item(),
        _h.min().item(),
        _w.max().item(),
        _h.max().item(),
    ]

    box = Bbox(whwh)
    # scale box to 1.05
    scale_box = box.scale(1.1, width=width, height=height)
    return scale_box

def query_model_name(model_name):
    if model_name in MODEL_PATH:
        model_path = MODEL_PATH[model_name]
        if not os.path.exists(model_path):
            model_url = MODEL_CARD[model_name]
            download_extract_tar_from_url(model_url, './')
    else:
        model_path = model_name
    
    return model_path


def query_model_config(model_name):
    try:
        model_params = model_name.split('-')[1]
        
        return MODEL_CONFIG[model_params] 
    except:
        return None

def infer_preprocess_image(
    rgb_path,
    mask,
    intr,
    pad_ratio,
    bg_color,
    max_tgt_size,
    aspect_standard,
    enlarge_ratio,
    render_tgt_size,
    multiply,
    need_mask=True,
):
    """inferece
    image, _, _ = preprocess_image(image_path, mask_path=None, intr=None, pad_ratio=0, bg_color=1.0,
                                        max_tgt_size=896, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                        render_tgt_size=source_size, multiply=14, need_mask=True)

    """

    rgb = np.array(Image.open(rgb_path))
    rgb_raw = rgb.copy()

    bbox = get_bbox(mask)
    bbox_list = bbox.get_box()

    rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
    mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]



    # h, w, _ = rgb.shape
    # if w == h:
    #     rgb = rgb[:, 1:]
    #     mask = mask[:, 1:]
    # if w > h:
    #     rgb = np.transpose(rgb, (1, 0, 2))
    #     mask = np.transpose(mask, (1, 0))



    h, w, _ = rgb.shape
    assert w < h
    cur_ratio = h / w
    scale_ratio = cur_ratio / aspect_standard


    target_w = int(min(w * scale_ratio, h))
    if target_w - w >0:
        offset_w = (target_w - w) // 2

        rgb = np.pad(
            rgb,
            ((0, 0), (offset_w, offset_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((0, 0), (offset_w, offset_w)),
            mode="constant",
            constant_values=0,
        )
    else:
        target_h = w * aspect_standard
        # print(int(target_h - h))
        offset_h = max(int(target_h - h), 0)

        rgb = np.pad(
            rgb,
            ((offset_h, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

        mask = np.pad(
            mask,
            ((offset_h, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    rgb = rgb / 255.0  # normalize to [0, 1]
    mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float32)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # resize to specific size require by preprocessor of smplx-estimator.
    rgb = resize_image_keepaspect_np(rgb, max_tgt_size)
    mask = resize_image_keepaspect_np(mask, max_tgt_size)

    # crop image to enlarge human area.
    rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
        rgb, mask, aspect_standard, enlarge_ratio
    )
    if intr is not None:
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

    # resize to render_tgt_size for training

    tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
        cur_hw=rgb.shape[:2],
        aspect_standard=aspect_standard,
        tgt_size=render_tgt_size,
        multiply=multiply,
    )

    rgb = cv2.resize(
        rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
    )

    if intr is not None:

        # ******************** Merge *********************** #
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        assert (
            abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
        ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert (
            abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
        ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"

        # ******************** Merge *********************** #
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = (
        torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    )  # [1, 1, H, W]
    return rgb, mask, intr


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    if "export_mesh" not in cli_cfg: 
        cli_cfg.export_mesh = None
    if "export_video" not in cli_cfg: 
        cli_cfg.export_video= None

    query_model = AutoModelQuery()

    # parse from ENV
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        model_name = query_model_name(os.environ.get("APP_MODEL_NAME"))
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")
    else:
        model_name = cli_cfg.model_name
        model_path= query_model.query(model_name) 
        cli_cfg.model_name = model_path 
    
    model_config = query_model_config(model_name)

    if model_config is not None:
        cfg_train = OmegaConf.load(model_config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path
        cfg.mesh_dump = os.path.join("exps", "meshs", _relative_path)  # output path

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train



class HumanLRMInferrer(Inferrer):

    EXP_TYPE: str = "human_lrm_sapdino_bh_sd3_5"
    # EXP_TYPE: str = "human_lrm_sd3"

    def __init__(self):
        super().__init__()

        self.cfg, cfg_train = parse_configs()

        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )  # logger function

        # if do not download prior model, we automatically download them.
        prior_check()

        self.facedetect = FaceDetector(
            "./pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd",
            device=avaliable_device(),
        )
        self.pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device=avaliable_device()
        )
        try:
            self.parsingnet = SAM2Seg()
        except:
            self.parsingnet = None 

        self.model: ModelHumanLRM = self._build_model(self.cfg).to(self.device)

        self.motion_dict = dict()

    def _build_model(self, cfg):
        from LHM.models import model_dict

        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])

        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model

    def _default_source_camera(
        self,
        dist_to_center: float = 2.0,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, -dist_to_center],
                    [0, 1, 0, 0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(
            canonical_camera_extrinsics, canonical_camera_intrinsics
        )
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(
        self,
        n_views: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(
            n_views=n_views, device=device
        )
        render_camera_intrinsics = (
            create_intrinsics(
                f=0.75,
                c=0.5,
                device=device,
            )
            .unsqueeze(0)
            .repeat(render_camera_extrinsics.shape[0], 1, 1)
        )
        render_cameras = build_camera_standard(
            render_camera_extrinsics, render_camera_intrinsics
        )
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)


    def crop_face_image(self, image_path, parsing_mask=None):

        if parsing_mask is not None:
            body_rgb = np.array(Image.open(image_path))
            body_rgb = body_rgb / 255.0  # normalize to [0, 1]
            body_mask = parsing_mask / 255.0
            body_mask = (body_mask > 0.5).astype(np.float32)
            body_rgb = body_rgb[:, :, :3] * body_mask[:, :, None] + 1.0 * (1 - body_mask[:, :, None])
            rgb = np.uint8(body_rgb*255)
        else:
            rgb = np.array(Image.open(image_path))

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        bbox = self.facedetect(rgb)
        head_rgb = rgb[:, int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        head_rgb = head_rgb.permute(1, 2, 0)
        head_rgb = head_rgb.cpu().numpy()
        return head_rgb

    @torch.no_grad()
    def parsing(self, img_path):

        parsing_out = self.parsingnet(img_path=img_path, bbox=None)

        alpha = (parsing_out.masks * 255).astype(np.uint8)

        return alpha

    def infer(self, image_path, dump_mask_dir, save_body_path, save_face_path, dump_mesh_dir):


        shape_pose = self.pose_estimator(image_path)

        if shape_pose.ratio <= 0.1:
            print(f"body ratio is too small: {shape_pose.ratio}, continue")
            assert False

        source_size = self.cfg.source_size
        aspect_standard = 5.0 / 3

        try:
            parsing_mask = self.parsing(image_path)
        except:
            print("Error in parsing!")
            assert False
        else:
            print(f"Succeed in parsing! {dump_mask_dir}")

        # prepare reference image
        image, _, _ = infer_preprocess_image(
            image_path,
            mask=parsing_mask,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=896,
            aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=source_size,
            multiply=14,
            need_mask=True,
        )
        image = image[0]
        body_rgb = np.array(Image.open(image_path))
        body_rgb = body_rgb / 255.0  # normalize to [0, 1]
        body_mask = parsing_mask / 255.0
        body_mask = (body_mask > 0.5).astype(np.float32)
        body_rgb = body_rgb[:, :, :3] * body_mask[:, :, None] + 1.0 * (1 - body_mask[:, :, None])
        Image.fromarray(np.uint8(body_rgb*255)).save(save_body_path)
        have_head = 1
        try:
            src_head_rgb = self.crop_face_image(image_path, parsing_mask)
            Image.fromarray(src_head_rgb).save(save_face_path)
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.cfg.src_head_size, self.cfg.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
        except:
            have_head = 0
            print("w/o head input!")
            src_head_rgb = np.zeros((112, 112, 3), dtype=np.uint8)
            src_head_rgb = np.zeros(
                (self.cfg.src_head_size, self.cfg.src_head_size, 3), dtype=np.uint8
            )
        src_head_rgb = (
            torch.from_numpy(src_head_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)
        )  # [1, 3, H, W]

        # vis_ref_img = (image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
        #     np.uint8
        # )
        # Image.fromarray(vis_ref_img).save(save_body_path)


        device = "cuda"
        dtype = torch.float32
        shape_param = torch.tensor(shape_pose.beta, dtype=dtype).unsqueeze(0)

        smplx_params =  dict()
        # cano pose setting
        smplx_params['betas'] = shape_param.to(device)

        smplx_params['root_pose'] = torch.zeros(1,1,3).to(device)
        smplx_params['body_pose'] = torch.zeros(1,1,21, 3).to(device)
        smplx_params['jaw_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['leye_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['reye_pose'] = torch.zeros(1, 1, 3).to(device)
        smplx_params['lhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
        smplx_params['rhand_pose'] = torch.zeros(1, 1, 15, 3).to(device)
        smplx_params['expr'] = torch.zeros(1, 1, 100).to(device)
        smplx_params['trans'] = torch.zeros(1, 1, 3).to(device)

        self.model.to(dtype)

        gs_app_model_list, query_points, transform_mat_neutral_pose = self.model.infer_single_view(
            image.unsqueeze(0).unsqueeze(0).to(device, dtype),
            src_head_rgb.unsqueeze(0).to(device, dtype),
            None,
            None,
            None,
            None,
            None,
            smplx_params={
                k: v.to(device) for k, v in smplx_params.items()
            },
        )
        smplx_params['transform_mat_neutral_pose'] = transform_mat_neutral_pose

        torch.save([
            gs_app_model_list[0].offset_xyz.squeeze().detach().cpu(), 
            query_points.squeeze().detach().cpu(), 
            gs_app_model_list[0].shs.squeeze().detach().cpu(), 
            gs_app_model_list[0].opacity.detach().cpu(), 
            gs_app_model_list[0].scaling.detach().cpu(), 
            gs_app_model_list[0].rotation.detach().cpu(), 
            transform_mat_neutral_pose.squeeze().detach().cpu(), 
            shape_param.squeeze().detach().cpu(), 
            shape_pose.ratio, 
            have_head
            ], dump_mesh_dir)
