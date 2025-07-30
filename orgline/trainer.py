#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OrgLine 
@IDE     ：PyCharm 
@Author  ：Alex Deng
@Date    ：2025/7/8 
'''

import os
import cv2
import numpy as np
import random
import warnings
import torch
import gc
from tqdm import tqdm


# Set environment variables and warning filters
def setup_environment():
    """Set up the training environment."""
    os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Ignore specific warnings
    warnings.filterwarnings("ignore", message="Overwriting .* in registry")
    warnings.filterwarnings("ignore", message="Argument(s) .* are not valid for transform")
    warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case of Affine transform")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", message=".*Flash attention.*")
    warnings.filterwarnings("ignore", message=".*Memory efficient.*")
    warnings.filterwarnings("ignore", message=".*CuDNN.*")
    warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass


def get_device(device_str=None):
    """
    Get the appropriate device for training.

    Args:
        device_str (str): Device string ('cuda:0', 'cuda:1', 'cpu', 'auto', None)

    Returns:
        torch.device: The selected device
    """
    if device_str is None or device_str.lower() == 'auto':
        if torch.cuda.is_available():
            # Automatically select the GPU with most free memory
            if torch.cuda.device_count() > 1:
                max_free_memory = 0
                best_device = 0
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_device = i
                device = torch.device(f'cuda:{best_device}')
                print(f"Auto-selected device: {device} (most free memory: {max_free_memory / 1024 ** 3:.2f}GB)")
            else:
                device = torch.device('cuda:0')
                print(f"Auto-selected device: {device}")
        else:
            device = torch.device('cpu')
            print(f"CUDA not available, using CPU")
    else:
        device = torch.device(device_str.lower())
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                device = torch.device('cpu')
            elif device.index is not None and device.index >= torch.cuda.device_count():
                print(f"Warning: GPU {device.index} not available, using cuda:0")
                device = torch.device('cuda:0')
        print(f"Selected device: {device}")

    return device


class SAM2Trainer:
    """Trainer for the SAM2 model."""

    def __init__(self, predictor, device=None):
        """
        Initialize the trainer.

        Args:
            predictor: The SAM2 predictor object.
            device: Device to use for training ('cuda:0', 'cuda:1', 'cpu', 'auto', or None)
        """
        self.device = get_device(device)
        self.predictor = predictor

        # Move model to selected device
        self.predictor.model = self.predictor.model.to(self.device)

        self.setup_model_training_mode()

        # Training configuration
        self.config = {
            'num_epochs': 30,
            'lr': 1e-4,
            'weight_decay': 4e-5,
            'gamma': 0.995,
            'patience': 500,
            'validation_samples': 50,
            'checkpoint_every': 1000,
            'validate_every': 500,
            'cleanup_every': 30,
            'accumulate_steps': 8
        }

        # Training state
        self.best_val_iou = 0
        self.step_count = 0
        self.mean_iou = 0

    def setup_model_training_mode(self):
        """Set the model training/eval mode appropriately."""
        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.image_encoder.train(False)
        self.predictor.model.mask_downsample.train(False)
        self.predictor.model.memory_encoder.train(False)
        self.predictor.model.memory_attention.train(False)
        self.predictor.model.sam_prompt_encoder.train(False)

    def clear_sam_cache(self):
        """Clear SAM2 feature cache."""
        try:
            if hasattr(self.predictor, '_features'):
                self.predictor._features = {}
            if hasattr(self.predictor, '_orig_hw'):
                self.predictor._orig_hw = []
            if hasattr(self.predictor, '_is_image_set'):
                self.predictor._is_image_set = False

            if hasattr(self.predictor, 'reset_image'):
                self.predictor.reset_image()
            elif hasattr(self.predictor, 'reset'):
                self.predictor.reset()

        except Exception as e:
            print(f"Warning: Could not clear SAM cache: {e}")

    def get_gpu_memory(self):
        """Get GPU memory usage."""
        if self.device.type == 'cuda' and torch.cuda.is_available():
            with torch.cuda.device(self.device):
                allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3
                cached = torch.cuda.memory_reserved(self.device) / 1024 ** 3
                return allocated, cached
        return 0, 0

    def cleanup_memory(self):
        """Clean up memory based on device type."""
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        gc.collect()

    def train_one_step(self, sample, optimizer, scaler, accumulate_steps=1):
        """Train a single step."""
        if sample is None or not sample['valid']:
            return None, None, None

        image = sample['image']
        masks = sample['masks']
        boxes = sample['boxes']

        if len(masks) == 0:
            return None, None, None

        variables_to_delete = []

        try:
            # Use autocast only for CUDA devices
            autocast_context = torch.cuda.amp.autocast() if self.device.type == 'cuda' else torch.no_grad()

            with autocast_context:
                self.predictor.set_image(image)

                mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    mask_logits=None,
                    normalize_coords=True
                )
                variables_to_delete.extend([mask_input, unnorm_coords, labels])

                sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                    points=None,
                    boxes=unnorm_box,
                    masks=None
                )
                variables_to_delete.extend([sparse_embeddings, dense_embeddings])

                batched_mode = unnorm_box.shape[0] > 1
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                     self.predictor._features["high_res_feats"]]
                variables_to_delete.append(high_res_features)

                low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                    image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                variables_to_delete.extend([low_res_masks, prd_scores])

                prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])
                variables_to_delete.append(prd_masks)

                gt_mask = torch.tensor(masks.astype(np.float32)).to(self.device)
                prd_mask = torch.sigmoid(prd_masks[:, 0])

                if gt_mask.shape != prd_mask.shape:
                    prd_mask = torch.nn.functional.interpolate(
                        prd_mask.unsqueeze(1),
                        size=gt_mask.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)

                variables_to_delete.extend([gt_mask, prd_mask])

                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-8) -
                            (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-8)).mean()

                prd_binary = (prd_mask > 0.5).float()
                inter = (gt_mask * prd_binary).sum(dim=(1, 2))
                union = gt_mask.sum(dim=(1, 2)) + prd_binary.sum(dim=(1, 2)) - inter
                iou = inter / (union + 1e-8)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                total_loss = seg_loss + score_loss * 0.05
                total_loss = total_loss / accumulate_steps

                variables_to_delete.extend([seg_loss, score_loss, prd_binary, inter, union, iou])

                current_iou = torch.mean(iou).item()
                loss_value = total_loss.item() * accumulate_steps

            try:
                if self.device.type == 'cuda':
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
            except Exception as e:
                print(f"Backpropagation error: {e}")
                return None, None, None

            return loss_value, current_iou, len(masks)

        except Exception as e:
            print(f"Training step error: {e}")
            return None, None, None

        finally:
            for var in variables_to_delete:
                try:
                    if isinstance(var, torch.Tensor):
                        del var
                    elif isinstance(var, list):
                        for item in var:
                            if isinstance(item, torch.Tensor):
                                del item
                        del var
                except:
                    pass

            self.clear_sam_cache()

    def train_batch_with_accumulation(self, batch_samples, optimizer, scaler, accumulate_steps):
        """Train a batch with gradient accumulation."""
        if not batch_samples:
            return None, None, None

        try:
            self.predictor.model.zero_grad()
        except:
            pass

        total_loss = 0
        total_iou = 0
        total_objects = 0
        valid_samples = 0

        for i, sample in enumerate(batch_samples):
            loss, iou, num_objects = self.train_one_step(
                sample, optimizer, scaler, accumulate_steps
            )

            if loss is not None:
                total_loss += loss
                total_iou += iou
                total_objects += num_objects
                valid_samples += 1

        if valid_samples > 0:
            try:
                if self.device.type == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                self.predictor.model.zero_grad()
            except Exception as e:
                print(f"Optimizer step error: {e}")
                self.predictor.model.zero_grad()
                return None, None, None

            avg_loss = total_loss / valid_samples
            avg_iou = total_iou / valid_samples

            return avg_loss, avg_iou, total_objects

        return None, None, None

    def validate_model(self, val_loader, num_samples=None):
        """Validate the model."""
        if num_samples is None:
            num_samples = self.config['validation_samples']

        self.predictor.model.eval()
        total_iou = 0
        total_samples = 0

        self.clear_sam_cache()
        self.cleanup_memory()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_samples:
                    break

                if batch is None:
                    continue

                samples = batch if isinstance(batch, list) else [batch]

                for sample in samples:
                    if not sample['valid']:
                        continue

                    image = sample['image']
                    masks = sample['masks']
                    boxes = sample['boxes']

                    if len(masks) == 0:
                        continue

                    try:
                        self.clear_sam_cache()

                        self.predictor.set_image(image)
                        masks_pred, scores, _ = self.predictor.predict(box=boxes, multimask_output=False)

                        if len(masks_pred) != len(masks):
                            min_len = min(len(masks_pred), len(masks))
                            masks_pred = masks_pred[:min_len]
                            masks = masks[:min_len]

                        sample_ious = []
                        for j in range(len(masks)):
                            try:
                                gt_mask = masks[j].astype(np.float32)
                                pred_mask = masks_pred[j].astype(np.float32)

                                if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                                    pred_mask = pred_mask.squeeze(0)
                                elif pred_mask.ndim == 3 and gt_mask.ndim == 2:
                                    pred_mask = pred_mask[0]

                                if gt_mask.shape != pred_mask.shape:
                                    pred_mask = cv2.resize(
                                        pred_mask,
                                        (gt_mask.shape[1], gt_mask.shape[0]),
                                        interpolation=cv2.INTER_NEAREST
                                    )
                                    pred_mask = pred_mask.astype(np.float32)

                                gt_mask = (gt_mask > 0.5).astype(np.float32)
                                pred_mask = (pred_mask > 0.5).astype(np.float32)

                                intersection = np.sum(gt_mask * pred_mask)
                                union = np.sum(gt_mask) + np.sum(pred_mask) - intersection

                                if union > 0:
                                    iou = intersection / union
                                    sample_ious.append(iou)
                                else:
                                    if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
                                        sample_ious.append(1.0)
                                    else:
                                        sample_ious.append(0.0)

                            except Exception as e:
                                print(f"Error computing IoU for mask {j}: {e}")
                                continue

                        if sample_ious:
                            total_iou += np.mean(sample_ious)
                            total_samples += 1

                        if total_samples % 10 == 0:
                            self.clear_sam_cache()
                            self.cleanup_memory()

                    except Exception as e:
                        print(f"Validation error at sample {total_samples}: {e}")
                        self.clear_sam_cache()
                        continue

        self.clear_sam_cache()
        self.cleanup_memory()
        self.predictor.model.train()

        return total_iou / total_samples if total_samples > 0 else 0

    def save_checkpoint(self, optimizer, scheduler, epoch, is_best=False, filepath=None):
        """Save a training checkpoint."""
        if filepath is None:
            filepath = 'best_model.torch' if is_best else f'checkpoint_step_{self.step_count}.torch'

        checkpoint = {
            'model_state_dict': self.predictor.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'step': self.step_count,
            'best_val_iou': self.best_val_iou,
            'mean_iou': self.mean_iou,
            'config': self.config,
            'device': str(self.device)
        }

        torch.save(checkpoint, filepath)

    def train(self, train_loader, val_loader, config_updates=None):
        """
        Main training function

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            config_updates: Dictionary of configuration updates (optional)

        Returns:
            dict: Training results
        """
        # Update configuration
        if config_updates:
            self.config.update(config_updates)

        # Set optimizer and scheduler
        optimizer = torch.optim.Adam(
            params=self.predictor.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['gamma'])

        # Initialize scaler only for CUDA devices
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        # Training parameters
        num_epochs = self.config['num_epochs']
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch

        # Training state
        no_improve = 0

        print(f"Starting training for {num_epochs} epochs ({total_steps} steps)")
        print(f"Training device: {self.device}")
        print(f"Configuration: {self.config}")

        # Initial GPU memory status
        allocated, cached = self.get_gpu_memory()
        if self.device.type == 'cuda':
            print(f"Initial GPU memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        else:
            print("Training on CPU")

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_iou = 0
            valid_batches = 0

            # Set current epoch for the dataset (if supported)
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)

            # Check data augmentation status
            if hasattr(train_loader.dataset, 'should_augment'):
                if train_loader.dataset.should_augment():
                    print(f"Epoch {epoch}: Using data augmentation")
                else:
                    print(f"Epoch {epoch}: Data augmentation disabled")

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=140)

            for batch_idx, batch_samples in enumerate(pbar):
                self.step_count += 1

                # Train one batch
                loss, iou, num_objects = self.train_batch_with_accumulation(
                    batch_samples, optimizer, scaler, self.config['accumulate_steps']
                )

                if loss is not None:
                    valid_batches += 1
                    epoch_loss += loss
                    epoch_iou += iou

                    # Update running mean IoU
                    if valid_batches == 1:
                        self.mean_iou = iou
                    else:
                        self.mean_iou = self.mean_iou * 0.99 + 0.01 * iou

                    # Monitor memory usage
                    if self.step_count % 20 == 0:
                        allocated, cached = self.get_gpu_memory()
                        if self.device.type == 'cuda' and allocated > 10:
                            print(f"\nWarning: High GPU memory usage: {allocated:.2f}GB")
                            self.cleanup_memory()

                    # Update progress bar
                    device_info = f"{allocated:.1f}GB" if self.device.type == 'cuda' else "CPU"
                    pbar.set_postfix({
                        "Loss": f"{loss:.4f}",
                        "IoU": f"{self.mean_iou:.3f}",
                        "Objs": num_objects,
                        "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "Device": device_info
                    })

                # Validation and checkpoint
                if self.step_count % self.config['validate_every'] == 0 and self.step_count > 0:
                    self.clear_sam_cache()
                    self.cleanup_memory()

                    val_iou = self.validate_model(val_loader)
                    print(f"\nStep {self.step_count}: Validation IoU = {val_iou:.4f}")

                    if val_iou > self.best_val_iou:
                        self.best_val_iou = val_iou
                        no_improve = 0

                        self.save_checkpoint(optimizer, scheduler, epoch, is_best=True)
                        print(f"New best model saved with IoU: {self.best_val_iou:.4f}")
                    else:
                        no_improve += self.config['validate_every']

                    if no_improve >= self.config['patience']:
                        print("Early stopping triggered")
                        return {
                            'best_val_iou': self.best_val_iou,
                            'final_epoch': epoch,
                            'total_steps': self.step_count,
                            'status': 'early_stopped',
                            'device': str(self.device)
                        }

                # Periodic checkpoint
                if self.step_count % self.config['checkpoint_every'] == 0:
                    self.save_checkpoint(optimizer, scheduler, epoch, is_best=False)

                # Memory cleanup
                if self.step_count % self.config['cleanup_every'] == 0:
                    self.clear_sam_cache()
                    self.cleanup_memory()

                scheduler.step()

            # Epoch summary
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                avg_iou = epoch_iou / valid_batches
                allocated, cached = self.get_gpu_memory()
                memory_info = f", GPU Memory = {allocated:.2f}GB" if self.device.type == 'cuda' else ""
                print(f"Epoch {epoch + 1} completed: Avg Loss = {avg_loss:.4f}, Avg IoU = {avg_iou:.4f}{memory_info}")

            # End-of-epoch cleanup
            self.clear_sam_cache()
            self.cleanup_memory()

        print("Training completed!")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")
        print(f"Training device: {self.device}")

        return {
            'best_val_iou': self.best_val_iou,
            'final_epoch': num_epochs,
            'total_steps': self.step_count,
            'status': 'completed',
            'device': str(self.device)
        }


def train_orglineseg_model(predictor, train_loader, val_loader, config=None, device=None):
    """
    Main interface function for training the SAM2 model.

    Args:
        predictor: SAM2 predictor object
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary (optional)
        device: Device to use ('cuda:0', 'cuda:1', 'cpu', 'auto', or None)

    Returns:
        dict: Training results
    """
    # Set up environment
    setup_environment()

    # Create trainer with device selection
    trainer = SAM2Trainer(predictor, device=device)

    # Start training
    results = trainer.train(train_loader, val_loader, config)

    return results


# Example usage
if __name__ == "__main__":
    # Create data loaders externally
    from train_data import create_dataloaders

    # Load dataset
    data_cls = ['Intestine', 'brain', 'colon', 'PDAC']
    data_name = data_cls[0]
    root_path = r'datasets'

    train_loader, val_loader = create_dataloaders(
        root_path="datasets",
        data_name="Intestine",
        type_T='train',
        batch_size=8,
        num_workers=0,
        image_size=512,
        max_instances=16,
        train_augment=True,
        train_size=0.8,
        use_albumentations=True,  # Use albumentations
        disable_augment_after_epoch=15
    )

    # Assume you have the predictor object
    # predictor = load_sam2_model(...)

    # Example 1: Use automatic device selection (default)
    results = train_orglineseg_model(
        predictor=predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        device='auto'  # Automatically select best available device
    )

    # Example 2: Use specific GPU
    results = train_orglineseg_model(
        predictor=predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda:0'  # Use first GPU
    )

    # Example 3: Use second GPU
    results = train_orglineseg_model(
        predictor=predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda:1'  # Use second GPU
    )

    # Example 4: Use CPU
    results = train_orglineseg_model(
        predictor=predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu'  # Use CPU
    )

    # Example 5: Use custom configuration with device selection
    custom_config = {
        'num_epochs': 50,
        'lr': 5e-5,
        'accumulate_steps': 4,
        'patience': 1000,
        "gamma": 0.995
    }

    results = train_orglineseg_model(
        predictor=predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        config=custom_config,
        device='cuda:0'  # Specify device
    )

    print(f"Training results: {results}")
    print(f"Training completed on device: {results['device']}")