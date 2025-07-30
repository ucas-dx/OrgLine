#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OrgLine 
@File    ：weights_merge.py
@IDE     ：PyCharm 
@Author  ：Alex Deng
@Date    ：2025/7/9 16:09 
'''
import os
import torch
from collections import OrderedDict


class ModelWeightMerger:
    """Model weight merging and loading utility class"""

    def __init__(self, verbose=True):
        self.verbose = verbose

    def _print(self, message):
        """Control print output"""
        if self.verbose:
            print(message)

    def check_files(self, lightweight_path, sam_path):
        """Check if required files exist"""
        self._print("=== Checking Required Files ===")

        for file_path in [lightweight_path, sam_path]:
            if not os.path.exists(file_path):
                self._print(f"❌ File not found: {file_path}")
                return False
            else:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                self._print(f"✅ {os.path.basename(file_path)} - {size_mb:.1f} MB")
        return True

    def load_and_merge_weights(self, lightweight_path, sam_path):
        """Load and merge weights"""
        self._print("\n=== Loading and Merging Weights ===")

        # 1. Load lightweight checkpoint
        self._print("1. Loading lightweight checkpoint...")
        try:
            checkpoint = torch.load(lightweight_path, map_location='cpu')
            self._print("✅ Lightweight weights loaded successfully")

            if not checkpoint.get('is_lightweight', False):
                self._print("⚠️  This is not a lightweight model file!")

            if self.verbose:
                self._print(f"Checkpoint keys: {list(checkpoint.keys())}")

        except Exception as e:
            self._print(f"❌ Failed to load lightweight weights: {e}")
            return None, None

        # 2. Load SAM2 weights
        self._print("\n2. Loading SAM2 weights...")
        try:
            sam_checkpoint = torch.load(sam_path, map_location='cpu')
            sam_state = sam_checkpoint['model']
            self._print("✅ SAM2 weights loaded successfully")
            self._print(f"SAM2 weight count: {len(sam_state)}")

        except Exception as e:
            self._print(f"❌ Failed to load SAM2 weights: {e}")
            return None, None

        # 3. Merge weights
        self._print("\n3. Merging weights...")
        complete_state = checkpoint['lightweight_state_dict'].copy()
        self._print(f"Lightweight weight count: {len(complete_state)}")

        # Add SAM2 weights (key mapping)
        transferred_count = 0
        for sam_key, sam_weight in sam_state.items():
            if sam_key.startswith('image_encoder.trunk.'):
                clean_key = sam_key.replace('image_encoder.trunk.', '')
                orgdet_key = f'model.0.model.{clean_key}'
                complete_state[orgdet_key] = sam_weight.clone()
                transferred_count += 1

        self._print(f"Transferred {transferred_count} weights from SAM2")
        self._print(f"Total weight count after merge: {len(complete_state)}")

        return checkpoint, complete_state

    def save_complete_checkpoint(self, temp_model, checkpoint, save_path='orgdet_complete_merged.pt'):
        """Save complete checkpoint with merged weights"""
        self._print(f"\n=== Saving Complete Checkpoint to {save_path} ===")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Build checkpoint dictionary
            weights_dict = {
                'model': temp_model,  # Directly save model object
                'epoch': checkpoint.get('epoch', 0),
                'best_fitness': checkpoint.get('best_fitness', 0.0),
                'ema': None,
                'updates': checkpoint.get('updates', 0),
                'optimizer': None,
                'train_args': checkpoint.get('train_args', {}),
                'date': checkpoint.get('date', None),
                'version': checkpoint.get('version', None),
                'merged_weights': True,  # Mark as merged weights
                'is_lightweight': False  # No longer lightweight after merging
            }

            # Save checkpoint
            torch.save(weights_dict, save_path)
            self._print(f"✅ Complete checkpoint saved to: {save_path}")

            # Check file size
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            self._print(f"File size: {file_size:.1f} MB")

            return True

        except Exception as e:
            self._print(f"❌ Save failed: {e}")
            return False

    def safe_load_weights_to_model(self, model, complete_state, strict=False):
        """Safely load weights to model"""
        self._print(f"\n=== Safely Loading Weights to Model (strict={strict}) ===")

        try:
            # Get underlying model
            temp_model = model.model
            self._print(f"Underlying model type: {type(temp_model)}")

            # Check weight matching
            model_keys = set(temp_model.state_dict().keys())
            state_keys = set(complete_state.keys())

            missing_keys = model_keys - state_keys
            unexpected_keys = state_keys - model_keys

            self._print(f"Model parameter count: {len(model_keys)}")
            self._print(f"Weight parameter count: {len(state_keys)}")
            self._print(f"Missing parameters: {len(missing_keys)}")
            self._print(f"Unexpected parameters: {len(unexpected_keys)}")

            if missing_keys and self.verbose:
                self._print("\nMissing parameters (first 5):")
                for key in list(missing_keys)[:5]:
                    self._print(f"  - {key}")

            if unexpected_keys and self.verbose:
                self._print("\nUnexpected parameters (first 5):")
                for key in list(unexpected_keys)[:5]:
                    self._print(f"  - {key}")

            # Load weights
            temp_model.load_state_dict(complete_state, strict=strict)
            self._print(f"✅ Weights loaded successfully")

            return temp_model

        except Exception as e:
            self._print(f"❌ Failed to load weights: {e}")

            # If strict=True fails, try strict=False
            if strict:
                self._print("Trying non-strict mode...")
                return self.safe_load_weights_to_model(model, complete_state, strict=False)

            # If still fails, try cleaning weight key names
            self._print("Trying to clean weight key names...")
            try:
                cleaned_state = OrderedDict()
                for key, value in complete_state.items():
                    new_key = key
                    # Remove possible prefixes
                    prefixes_to_remove = ['model.0.', 'model.', 'module.']
                    for prefix in prefixes_to_remove:
                        if new_key.startswith(prefix):
                            new_key = new_key[len(prefix):]
                            break
                    cleaned_state[new_key] = value

                temp_model = model.model
                temp_model.load_state_dict(cleaned_state, strict=False)
                self._print("✅ Weight loading successful after cleaning key names")
                return temp_model

            except Exception as e2:
                self._print(f"❌ Still failed after cleaning key names: {e2}")
                return None

    def merge_and_load_weights(self, model, lightweight_path, sam_path, save_path=None):
        """
        Main interface: Merge weights and load to model

        Args:
            model: Model object to load weights into
            lightweight_path: Path to lightweight weight file
            sam_path: Path to SAM2 weight file
            save_path: Path to save merged weights (optional)

        Returns:
            model: Model with loaded weights, None if failed
        """
        try:
            # 1. Check files
            if not self.check_files(lightweight_path, sam_path):
                return None

            # 2. Load and merge weights
            checkpoint, complete_state = self.load_and_merge_weights(lightweight_path, sam_path)
            if checkpoint is None or complete_state is None:
                return None

            # 3. Load weights to model
            temp_model = self.safe_load_weights_to_model(model, complete_state)
            if temp_model is None:
                return None

            # 4. Save complete checkpoint if save_path is provided
            if save_path:
                save_success = self.save_complete_checkpoint(temp_model, checkpoint, save_path)
                if not save_success:
                    self._print("⚠️  Continuing despite save failure...")

            # 5. Update model
            model.model = temp_model

            # 6. Set evaluation mode
            if hasattr(model, 'eval'):
                model.eval()

            self._print("🎉 Model weight merging and loading completed!")

            # 7. Simple verification
            total_params = sum(p.numel() for p in model.parameters())
            self._print(f"Total model parameters: {total_params:,}")

            return model

        except Exception as e:
            self._print(f"❌ Weight merging and loading failed: {e}")
            return None


def merge_and_load_model_weights(model, lightweight_path, sam_path, save_path=None, verbose=True):
    """
    Convenience function: Merge SAM2 and lightweight weights and load to model

    Args:
        model: Model object to load weights into (e.g., YOLO model)
        lightweight_path (str): Path to lightweight weight file (e.g., 'orgdet.pt')
        sam_path (str): Path to SAM2 weight file (e.g., 'sam2_hiera_large.pt')
        save_path (str): Path to save merged weights (optional, e.g., 'orgdet_complete_merged.pt')
        verbose (bool): Whether to show detailed information, default True

    Returns:
        model: Model object with loaded weights, None if failed

    Example:
        from ultralytics import YOLO

        # Initialize model
        model = YOLO('path/to/config.yaml')

        # Merge and load weights, and save merged checkpoint
        loaded_model = merge_and_load_model_weights(
            model=model,
            lightweight_path='orgdet.pt',
            sam_path='sam2_hiera_large.pt',
            save_path='orgdet_complete_merged.pt',  # Save complete merged checkpoint
            verbose=True
        )

        if loaded_model is not None:
            print("Model loaded successfully!")
            # Now you can use loaded_model for inference
        else:
            print("Model loading failed!")
    """
    merger = ModelWeightMerger(verbose=verbose)
    return merger.merge_and_load_weights(model, lightweight_path, sam_path, save_path)


# Usage example
if __name__ == "__main__":
    # Example usage
    print("=== Model Weight Merging Tool ===")
    print("\nUsage:")
    print("1. Prepare lightweight weight file and SAM2 weight file")
    print("2. Initialize your model")
    print("3. Call merge_and_load_model_weights function")
    print("\nExample code:")
    print("from ultralytics import YOLO")
    print("model = YOLO('config.yaml')")
    print("loaded_model = merge_and_load_model_weights(")
    print("    model=model,")
    print("    lightweight_path='orgdet.pt',")
    print("    sam_path='sam2_hiera_large.pt',")
    print("    save_path='orgdet_complete_merged.pt'  # Save complete checkpoint")

    print(")")