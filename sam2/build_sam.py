# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


# def build_sam2(
#     config_file,
#     ckpt_path=None,
#     device="cuda",
#     mode="eval",
#     hydra_overrides_extra=[],
#     apply_postprocessing=True,
#     **kwargs,
# ):
#
#     if apply_postprocessing:
#         hydra_overrides_extra = hydra_overrides_extra.copy()
#         hydra_overrides_extra += [
#             # dynamically fall back to multi-mask if the single mask is not stable
#             "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
#             "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
#             "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
#         ]
#     # Read config and init model
#     cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
#     OmegaConf.resolve(cfg)
#     model = instantiate(cfg.model, _recursive_=True)
#     _load_checkpoint(model, ckpt_path)
#     model = model.to(device)
#     if mode == "eval":
#         model.eval()
#     return model


import logging
import os
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
        config_file,
        ckpt_path=None,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
        config_dir=None,
        **kwargs,
):
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    # 清除之前的Hydra实例
    GlobalHydra.instance().clear()

    # 确定配置目录
    if config_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(current_dir, "configs")

    config_dir = os.path.abspath(config_dir)

    # 确保配置目录存在
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # 检查配置文件是否存在
    config_file_path = os.path.join(config_dir, config_file)
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    try:
        # 使用配置目录初始化Hydra
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    except Exception as e:
        logging.error(f"Failed to load config with Hydra: {e}")
        # 备用方案：直接加载YAML文件
        cfg = OmegaConf.load(config_file_path)
        # 手动应用覆盖参数
        for override in hydra_overrides_extra:
            if override.startswith("++"):
                key_value = override[2:].split("=", 1)
                if len(key_value) == 2:
                    key, value = key_value
                    OmegaConf.set(cfg, key, value)

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
