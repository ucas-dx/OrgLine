#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OrgLine 
@IDE     ：PyCharm 
@Author  ：Alex Deng
@Date    ：2025/7/8 
'''

import numpy as np
import torch
import cv2
import os
import urllib.request
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2ModelLoader:
    """SAM2 Model Loader for automatic downloading and loading"""

    # Predefined model configurations
    MODEL_CONFIGS = {
        'hiera_large': {
            'checkpoint': 'sam2_hiera_large.pt',
            'config': 'sam2_hiera_l.yaml',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt',
            'min_size': 1024 * 1024  # 1MB
        },
        'hiera_base_plus': {
            'checkpoint': 'sam2_hiera_base_plus.pt',
            'config': 'sam2_hiera_b+.yaml',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
            'min_size': 1024 * 1024
        },
        'hiera_small': {
            'checkpoint': 'sam2_hiera_small.pt',
            'config': 'sam2_hiera_s.yaml',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt',
            'min_size': 1024 * 1024
        },
        'hiera_tiny': {
            'checkpoint': 'sam2_hiera_tiny.pt',
            'config': 'sam2_hiera_t.yaml',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt',
            'min_size': 1024 * 1024
        }
    }

    def __init__(self, device="cuda", download_dir="./checkpoints"):
        """
        Initialize SAM2 Model Loader

        Args:
            device: Device type ("cuda" or "cpu")
            download_dir: Directory to store downloaded checkpoints
        """
        self.device = device
        self.download_dir = download_dir

    def download_with_progress(self, url, filename):
        """Download function with progress bar"""

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(
                    f"\rDownload progress: {percent}% ({downloaded // (1024 * 1024)} MB / {total_size // (1024 * 1024)} MB)",
                    end='', flush=True)

        try:
            # Create directory only if filename contains a directory path
            dirname = os.path.dirname(filename)
            if dirname:  # Only create directory if dirname is not empty
                os.makedirs(dirname, exist_ok=True)

            # 使用临时文件下载，完成后再重命名
            temp_filename = filename + '.tmp'

            # 清理可能存在的临时文件
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            print(f"Downloading to temporary file: {temp_filename}")
            urllib.request.urlretrieve(url, temp_filename, progress_hook)

            # 下载完成后重命名
            if os.path.exists(temp_filename):
                os.rename(temp_filename, filename)
                print(f"\nDownload completed: {filename}")
                return True
            else:
                print(f"\nError: Temporary file not found after download")
                return False

        except Exception as e:
            print(f"\nDownload failed: {e}")
            # 清理临时文件
            temp_filename = filename + '.tmp'
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass
            return False

    def verify_checkpoint(self, checkpoint_path, min_size):
        """Verify checkpoint file integrity"""
        if not os.path.exists(checkpoint_path):
            print(f"File does not exist: {checkpoint_path}")
            return False

        try:
            file_size = os.path.getsize(checkpoint_path)
            print(f"Found existing file: {checkpoint_path} (size: {file_size / (1024 * 1024):.2f} MB)")

            # 更宽松的大小检查 - 只要文件大于100MB就认为可能是有效的
            if file_size < 100 * 1024 * 1024:  # 100MB minimum for SAM2 models
                print(f"Warning: File seems too small for a SAM2 model (size: {file_size / (1024 * 1024):.2f} MB)")
                return False

            # 尝试加载文件头部来验证这是一个PyTorch模型文件
            try:
                import torch
                # 只检查文件是否可以被torch识别，不完全加载
                torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                print("File appears to be a valid PyTorch checkpoint")
                return True
            except Exception as e:
                print(f"File exists but appears corrupted (PyTorch load test failed): {e}")
                return False

        except Exception as e:
            print(f"Error verifying file: {e}")
            return False

    def download_checkpoint(self, url, checkpoint_path, min_size):
        """Download checkpoint file with verification"""
        print(f"Checking for existing file: {checkpoint_path}")

        # Check if file exists and is complete
        if self.verify_checkpoint(checkpoint_path, min_size):
            print(f"✓ Valid checkpoint file found, skipping download")
            return True

        # File doesn't exist or is incomplete, need to download
        if os.path.exists(checkpoint_path):
            print("Removing potentially corrupted file...")
            try:
                os.remove(checkpoint_path)
            except Exception as e:
                print(f"Warning: Could not remove existing file: {e}")

        print(f"Starting download from: {url}")
        print(f"Saving to: {checkpoint_path}")

        # 添加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            print(f"Download attempt {attempt + 1}/{max_retries}")
            success = self.download_with_progress(url, checkpoint_path)

            if success:
                # Verify downloaded file
                if self.verify_checkpoint(checkpoint_path, min_size):
                    print("✓ Download completed and verified successfully")
                    return True
                else:
                    print(f"Downloaded file verification failed (attempt {attempt + 1})")
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
            else:
                print(f"Download failed (attempt {attempt + 1})")

            if attempt < max_retries - 1:
                print("Retrying download...")

        print("✗ All download attempts failed")
        return False

    def load_model(self, checkpoint_path, model_config, config_dir, download_url=None, min_size=1024 * 1024):
        """
        Load SAM2 model

        Args:
            checkpoint_path: Path to checkpoint file
            model_config: Model configuration file name
            config_dir: Configuration files directory
            download_url: Download URL (optional)
            min_size: Minimum file size for verification

        Returns:
            SAM2ImagePredictor: Predictor object
        """
        # If checkpoint_path is just a filename, prepend download_dir
        if not os.path.dirname(checkpoint_path):
            checkpoint_path = os.path.join(self.download_dir, checkpoint_path)

        # 规范化路径，避免路径不一致导致的问题
        checkpoint_path = os.path.abspath(checkpoint_path)

        # Create download directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        print(f"Target checkpoint path: {checkpoint_path}")

        # If download URL is provided, attempt to download
        if download_url:
            if not self.download_checkpoint(download_url, checkpoint_path, min_size):
                raise FileNotFoundError(f"Unable to download or verify weight file: {checkpoint_path}")
        else:
            # Only verify file existence
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Weight file {checkpoint_path} does not exist")

        # Load model
        try:
            print("Loading model...")
            sam2_model = build_sam2(model_config, checkpoint_path, config_dir=config_dir, device=self.device)
            predictor = SAM2ImagePredictor(sam2_model)
            print("Model loaded successfully!")
            return predictor

        except Exception as e:
            print(f"Model loading failed: {e}")

            # If download URL exists, try to re-download
            if download_url:
                print("Attempting to re-download weight file...")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)

                if self.download_checkpoint(download_url, checkpoint_path, min_size):
                    try:
                        sam2_model = build_sam2(model_config, checkpoint_path, config_dir=config_dir,
                                                device=self.device)
                        predictor = SAM2ImagePredictor(sam2_model)
                        print("Model loaded successfully after re-download!")
                        return predictor
                    except Exception as e2:
                        raise RuntimeError(f"Model loading failed even after re-download: {e2}")
                else:
                    raise RuntimeError(f"Failed to re-download weight file: {checkpoint_path}")
            else:
                raise RuntimeError(f"Model loading failed: {e}")

    def load_pretrained_model(self, model_name, config_dir):
        """
        Load pretrained model

        Args:
            model_name: Model name ('hiera_large', 'hiera_base_plus', 'hiera_small', 'hiera_tiny')
            config_dir: Configuration files directory

        Returns:
            SAM2ImagePredictor: Predictor object
        """
        if model_name not in self.MODEL_CONFIGS:
            available_models = list(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown model name: {model_name}. Available models: {available_models}")

        config = self.MODEL_CONFIGS[model_name]

        return self.load_model(
            checkpoint_path=config['checkpoint'],
            model_config=config['config'],
            config_dir=config_dir,
            download_url=config['url'],
            min_size=config['min_size']
        )


def load_sam2_model(checkpoint_path, model_config, config_dir, download_url=None, device="cuda",
                    download_dir="./checkpoints"):
    """
    Simplified SAM2 model loading interface

    Args:
        checkpoint_path: Path to checkpoint file
        model_config: Model configuration file name
        config_dir: Configuration files directory
        download_url: Download URL (optional)
        device: Device type
        download_dir: Directory to store downloaded checkpoints

    Returns:
        SAM2ImagePredictor: Predictor object
    """
    loader = SAM2ModelLoader(device=device, download_dir=download_dir)
    return loader.load_model(checkpoint_path, model_config, config_dir, download_url)


def load_sam2_pretrained(model_name="hiera_large", config_dir=r'orgline/sam2_configs', device="cuda",
                         download_dir="./checkpoints"):
    """
    Convenient interface for loading pretrained SAM2 models

    Args:
        model_name: Model name ('hiera_large', 'hiera_base_plus', 'hiera_small', 'hiera_tiny')
        config_dir: Configuration files directory
        device: Device type
        download_dir: Directory to store downloaded checkpoints

    Returns:
        SAM2ImagePredictor: Predictor object
    """
    loader = SAM2ModelLoader(device=device, download_dir=download_dir)
    return loader.load_pretrained_model(model_name, config_dir)


# Usage examples
if __name__ == "__main__":
    # Method 1: Using custom parameters
    # predictor = load_sam2_model(
    #     checkpoint_path="sam2_hiera_large.pt",
    #     model_config="sam2_hiera_l.yaml",
    #     config_dir=r'orgline/sam2_configs',
    #     download_url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    #     download_dir="./checkpoints"  # Specify download directory
    # )

    # Method 2: Using pretrained model (recommended)
    predictor = load_sam2_pretrained(
        model_name="hiera_large",
        config_dir=r'sam2_configs',
        download_dir="../"  # Specify download directory
    )

    # # Method 3: Using other pretrained models
    # predictor = load_sam2_pretrained(
    #     model_name="hiera_base_plus",
    #     config_dir=r'orgline/sam2_configs',
    #     download_dir="./checkpoints"  # Specify download directory
    # )
    #
    # print("Model ready for use!")