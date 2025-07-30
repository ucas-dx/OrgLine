#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OrgLine 
@IDE     ：PyCharm 
@Author  ：Alex Deng
@Date    ：2025/7/8 
Enhanced with TIFF, CZI, ND2 support
'''

import os
import cv2
import numpy as np
import random
import warnings
from torch.utils.data import Dataset, DataLoader

# Ignore specific warnings
warnings.filterwarnings("ignore", message="Overwriting .* in registry")
warnings.filterwarnings("ignore", message="Argument(s) .* are not valid for transform")
warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case of Affine transform")

# Try importing albumentations
try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True

    # Get albumentations version for API compatibility
    try:
        import albumentations as A

        AL_VERSION = A.__version__
        AL_MAJOR_VERSION = int(AL_VERSION.split('.')[0])
        AL_MINOR_VERSION = int(AL_VERSION.split('.')[1])
    except:
        AL_MAJOR_VERSION = 1
        AL_MINOR_VERSION = 0

except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    AL_MAJOR_VERSION = 0
    AL_MINOR_VERSION = 0
    print("Warning: albumentations not installed. Using basic augmentation only.")

# Try importing scientific image format libraries
try:
    import tifffile

    TIFFFILE_AVAILABLE = True
    print("tifffile available for enhanced TIFF support")
except ImportError:
    TIFFFILE_AVAILABLE = False
    print("Warning: tifffile not installed. Using OpenCV for TIFF files.")

try:
    from czifile import imread as czi_imread

    CZI_AVAILABLE = True
    print("czifile available for CZI support")
except ImportError:
    CZI_AVAILABLE = False
    print("Warning: czifile not installed. CZI files not supported.")

try:
    import nd2

    ND2_AVAILABLE = True
    print("nd2 available for ND2 support")
except ImportError:
    ND2_AVAILABLE = False
    try:
        from nd2reader import ND2Reader

        ND2_READER_AVAILABLE = True
        ND2_AVAILABLE = True
        print("nd2reader available for ND2 support")
    except ImportError:
        ND2_READER_AVAILABLE = False
        print("Warning: nd2/nd2reader not installed. ND2 files not supported.")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available for additional image format support.")


def load_scientific_image(image_path):
    """
    Load scientific image formats with appropriate libraries.

    Args:
        image_path (str): Path to the image file

    Returns:
        numpy.ndarray: Image array in RGB format (H, W, C) or grayscale (H, W)
    """
    file_ext = os.path.splitext(image_path)[1].lower()

    try:
        if file_ext in ['.tif', '.tiff']:
            return load_tiff_image(image_path)
        elif file_ext == '.czi':
            return load_czi_image(image_path)
        elif file_ext == '.nd2':
            return load_nd2_image(image_path)
        else:
            # Use standard OpenCV for other formats
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Failed to load image: {image_path}")

    except Exception as e:
        print(f"Error loading {image_path} with specialized loader: {e}")
        # Fallback to OpenCV
        try:
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Failed to load image with fallback: {image_path}")
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")
            raise


def load_tiff_image(image_path):
    """Load TIFF images with enhanced support."""
    if TIFFFILE_AVAILABLE:
        try:
            # Use tifffile for better TIFF support
            image = tifffile.imread(image_path)

            # Handle different TIFF formats
            if len(image.shape) == 2:
                # Grayscale
                return image
            elif len(image.shape) == 3:
                # Multi-channel image
                if image.shape[2] == 3:
                    # RGB
                    return image
                elif image.shape[2] == 4:
                    # RGBA, convert to RGB
                    return image[:, :, :3]
                elif image.shape[2] > 4:
                    # Multi-channel, take first 3 channels
                    return image[:, :, :3]
                else:
                    # Convert to 3-channel
                    return np.stack([image[:, :, 0]] * 3, axis=2)
            elif len(image.shape) == 4:
                # Multi-page TIFF, take first page
                first_page = image[0]
                if len(first_page.shape) == 2:
                    return np.stack([first_page] * 3, axis=2)
                else:
                    return first_page[:, :, :3] if first_page.shape[2] >= 3 else first_page
            else:
                raise ValueError(f"Unsupported TIFF shape: {image.shape}")

        except Exception as e:
            print(f"tifffile failed for {image_path}: {e}")
            # Fallback to OpenCV
            pass

    # Fallback to OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot load TIFF file: {image_path}")

    if len(image.shape) == 2:
        # Grayscale to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            # Take first 3 channels
            return image[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def load_czi_image(image_path):
    """Load CZI (Carl Zeiss Image) files."""
    if not CZI_AVAILABLE:
        raise ValueError("CZI support not available. Install czifile: pip install czifile")

    try:
        # Load CZI file
        czi_data = czi_imread(image_path)

        # CZI files often have complex dimensions (T, Z, C, Y, X)
        # We need to extract a 2D image
        if len(czi_data.shape) == 2:
            # Already 2D
            image = czi_data
        elif len(czi_data.shape) == 3:
            # Could be (C, Y, X) or (Y, X, C)
            if czi_data.shape[0] <= 4:  # Assume first dim is channels
                image = np.transpose(czi_data, (1, 2, 0))
            else:
                image = czi_data
        elif len(czi_data.shape) > 3:
            # Multi-dimensional, take first slice of each dimension until we get 2D or 3D
            image = czi_data
            while len(image.shape) > 3:
                image = image[0]

            if len(image.shape) == 3 and image.shape[0] <= 4:
                image = np.transpose(image, (1, 2, 0))

        # Normalize if needed (CZI files might have high bit depth)
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=2)
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                return np.stack([image[:, :, 0]] * 3, axis=2)
            elif image.shape[2] >= 3:
                return image[:, :, :3]
            else:
                # Pad to 3 channels
                channels_needed = 3 - image.shape[2]
                padding = np.zeros((image.shape[0], image.shape[1], channels_needed), dtype=image.dtype)
                return np.concatenate([image, padding], axis=2)

        return image

    except Exception as e:
        raise ValueError(f"Failed to load CZI file {image_path}: {e}")


def load_nd2_image(image_path):
    """Load ND2 (Nikon) files."""
    if not ND2_AVAILABLE:
        raise ValueError("ND2 support not available. Install nd2 or nd2reader: pip install nd2")

    try:
        if 'nd2' in globals() and hasattr(nd2, 'ND2File'):
            # Use newer nd2 package
            with nd2.ND2File(image_path) as nd2_file:
                # Get image data (usually the first frame/channel)
                if len(nd2_file.shape) == 2:
                    image = nd2_file[:]
                elif len(nd2_file.shape) == 3:
                    # Could be (T, Y, X), (C, Y, X), or (Y, X, C)
                    if nd2_file.shape[0] <= 10:  # Assume first dim is time/channel
                        image = nd2_file[0]  # Take first frame/channel
                    else:
                        image = nd2_file[:]
                elif len(nd2_file.shape) == 4:
                    # (T, C, Y, X) or similar
                    image = nd2_file[0, 0]  # First time point, first channel
                else:
                    # Take first slice of each dimension until 2D
                    image = nd2_file[:]
                    while len(image.shape) > 2:
                        image = image[0]

        elif ND2_READER_AVAILABLE:
            # Use nd2reader package
            with ND2Reader(image_path) as nd2_file:
                # Get basic info
                if len(nd2_file) == 1:
                    image = nd2_file[0]
                else:
                    # Multiple frames, take first
                    image = nd2_file[0]
        else:
            raise ValueError("No ND2 reader available")

        # Normalize if needed
        if image.dtype != np.uint8:
            if image.max() > 255:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=2)
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                return np.stack([image[:, :, 0]] * 3, axis=2)
            elif image.shape[2] >= 3:
                return image[:, :, :3]
            else:
                # Pad to 3 channels
                channels_needed = 3 - image.shape[2]
                padding = np.zeros((image.shape[0], image.shape[1], channels_needed), dtype=image.dtype)
                return np.concatenate([image, padding], axis=2)

        return image

    except Exception as e:
        raise ValueError(f"Failed to load ND2 file {image_path}: {e}")


def ensure_contiguous(array):
    """Ensure the array is contiguous to avoid negative stride issues."""
    if array.strides[-1] < 0 or not array.flags['C_CONTIGUOUS']:
        return array.copy()
    return array


def filter_small_components(mask, min_area_ratio=0.5):
    """Filter out small disconnected components for each instance."""
    # Ensure mask is contiguous
    mask = ensure_contiguous(mask)

    filtered_mask = np.zeros_like(mask)
    unique_ids = np.unique(mask)[1:]  # exclude background (0)

    for instance_id in unique_ids:
        binary_mask = (mask == instance_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask)

        if num_labels > 1:
            component_areas = [(label, np.sum(labels == label)) for label in range(1, num_labels)]
            component_areas.sort(key=lambda x: x[1], reverse=True)

            if len(component_areas) > 1:
                largest_area = component_areas[0][1]
                for label, area in component_areas:
                    if area >= largest_area * min_area_ratio:
                        filtered_mask[labels == label] = instance_id
            else:
                filtered_mask[labels == component_areas[0][0]] = instance_id

    return filtered_mask


def create_albumentations_transforms():
    """Create appropriate transforms based on albumentations version."""
    if not ALBUMENTATIONS_AVAILABLE:
        return None, None

    # GaussNoise parameters depend on version
    if AL_MAJOR_VERSION >= 1 and AL_MINOR_VERSION >= 3:
        gauss_noise = A.GaussNoise(noise_scale_factor=0.1, p=1.0)
    else:
        try:
            gauss_noise = A.GaussNoise(var_limit=(10, 50), p=1.0)
        except:
            gauss_noise = A.GaussNoise(p=1.0)

    # Geometric transforms (affect image & mask)
    geometric_transform = A.Compose([
        A.OneOf([
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT),
        ], p=0.5),

        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ], p=0.5),

        A.Affine(
            translate_percent=0.1,
            rotate=(-10, 10),
            shear=(-5, 5),
            scale=(0.9, 1.1),
            p=0.4
        ),

        A.ElasticTransform(
            alpha=1,
            sigma=50,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),

        A.Perspective(scale=(0.05, 0.1), p=0.2),
    ])

    # Color transforms (image only)
    color_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.5),

        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
        ], p=0.3),

        A.OneOf([
            gauss_noise,
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
    ])

    return geometric_transform, color_transform


class SegmentationDataset(Dataset):
    """Custom Dataset for segmentation data with scientific image format support."""

    def __init__(self, data_list, max_size=512, augment=True, max_instances=20,
                 use_albumentations=True, disable_augment_after_epoch=None):
        self.data_list = data_list
        self.max_size = max_size
        self.augment = augment
        self.max_instances = max_instances
        self.use_albumentations = use_albumentations and ALBUMENTATIONS_AVAILABLE
        self.disable_augment_after_epoch = disable_augment_after_epoch
        self.current_epoch = 0  # Epoch counter

        # Initialize albumentations transforms if enabled
        if self.use_albumentations and self.augment:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.geometric_transform, self.color_transform = create_albumentations_transforms()
        else:
            self.geometric_transform = None
            self.color_transform = None

    def set_epoch(self, epoch):
        """Set the current epoch, for controlling augmentation."""
        self.current_epoch = epoch

    def should_augment(self):
        """Whether augmentation should be applied in the current epoch."""
        if not self.augment:
            return False

        if self.disable_augment_after_epoch is not None:
            return self.current_epoch < self.disable_augment_after_epoch

        return True

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get a single data sample."""
        ent = self.data_list[idx]

        # Load image and mask
        try:
            # Use specialized loading for scientific formats
            image = load_scientific_image(ent["image"])

            # Load mask (usually PNG, but could be TIFF)
            mask_path = ent["annotation"]
            mask_ext = os.path.splitext(mask_path)[1].lower()

            if mask_ext in ['.tif', '.tiff']:
                mask_img = load_tiff_image(mask_path)
                # Convert to grayscale if needed
                if len(mask_img.shape) == 3:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            else:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask_img is None:
                return self._get_empty_sample()

            # Ensure loaded data is contiguous
            image = ensure_contiguous(image)
            mask_img = ensure_contiguous(mask_img)

            # Resize image and mask
            r = min(self.max_size / image.shape[1], self.max_size / image.shape[0])
            new_width = int(image.shape[1] * r)
            new_height = int(image.shape[0] * r)

            image = cv2.resize(image, (new_width, new_height))
            mask_img = cv2.resize(mask_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            # Ensure resized data is contiguous
            image = ensure_contiguous(image)
            mask_img = ensure_contiguous(mask_img)

            # Apply data augmentation if enabled and should augment
            if self.should_augment():
                if self.use_albumentations:
                    image, mask_img = self._apply_albumentations(image, mask_img)
                else:
                    image, mask_img = self._apply_basic_augmentation(image, mask_img)

                # Ensure data after augmentation is contiguous
                image = ensure_contiguous(image)
                mask_img = ensure_contiguous(mask_img)

            # Filter small components
            filtered_mask = filter_small_components(mask_img)

            # Get instances
            instance_ids = np.unique(filtered_mask)[1:]  # exclude background

            if len(instance_ids) == 0:
                return self._get_empty_sample()

            # Limit number of instances by area (keep largest)
            if len(instance_ids) > self.max_instances:
                instance_areas = []
                for instance_id in instance_ids:
                    area = np.sum(filtered_mask == instance_id)
                    instance_areas.append((instance_id, area))

                instance_areas.sort(key=lambda x: x[1], reverse=True)
                instance_ids = [x[0] for x in instance_areas[:self.max_instances]]

            masks = []
            boxes = []

            for instance_id in instance_ids:
                mask = (filtered_mask == instance_id).astype(np.uint8)

                if np.sum(mask) == 0:
                    continue

                # Ensure mask is contiguous
                mask = ensure_contiguous(mask)
                masks.append(mask)

                # Calculate bounding box
                coords = np.argwhere(mask > 0)
                if len(coords) > 0:
                    ymin, xmin = coords.min(axis=0)
                    ymax, xmax = coords.max(axis=0)
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                    else:
                        masks.pop()
                        continue
                else:
                    masks.pop()
                    continue

            if len(masks) == 0:
                return self._get_empty_sample()

            # Ensure final arrays are contiguous
            image = ensure_contiguous(image)
            masks_array = np.array([ensure_contiguous(mask) for mask in masks])
            boxes_array = np.array(boxes)

            return {
                'image': image,
                'masks': masks_array,
                'boxes': boxes_array,
                'valid': True
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self._get_empty_sample()

    def _get_empty_sample(self):
        """Return an empty sample when data loading fails."""
        return {
            'image': np.zeros((100, 100, 3), dtype=np.uint8),
            'masks': np.array([]),
            'boxes': np.array([]),
            'valid': False
        }

    def _apply_albumentations(self, image, mask):
        """Apply albumentations augmentation."""
        try:
            # Ensure mask is uint8 and contiguous
            mask = ensure_contiguous(mask.astype(np.uint8))
            image = ensure_contiguous(image)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # First apply geometric transforms (image & mask together)
                if self.geometric_transform and random.random() < 0.7:
                    transformed = self.geometric_transform(image=image, mask=mask)
                    image = ensure_contiguous(transformed['image'])
                    mask = ensure_contiguous(transformed['mask'])

                # Then apply color transforms (image only)
                if self.color_transform and random.random() < 0.5:
                    transformed = self.color_transform(image=image)
                    image = ensure_contiguous(transformed['image'])

            return image, mask
        except Exception as e:
            print(f"Error in albumentations augmentation: {e}")
            # If albumentations fails, fallback to basic augmentation
            return self._apply_basic_augmentation(image, mask)

    def _apply_basic_augmentation(self, image, mask):
        """Apply basic data augmentation."""
        # Ensure input is contiguous
        image = ensure_contiguous(image)
        mask = ensure_contiguous(mask)

        # Random horizontal flip
        if random.random() < 0.5:
            image = ensure_contiguous(cv2.flip(image, 1))
            mask = ensure_contiguous(cv2.flip(mask, 1))

        # Random vertical flip
        if random.random() < 0.3:
            image = ensure_contiguous(cv2.flip(image, 0))
            mask = ensure_contiguous(cv2.flip(mask, 0))

        # Random brightness adjustment
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
            image = ensure_contiguous(image)

        # Random contrast adjustment
        if random.random() < 0.3:
            contrast = random.uniform(0.8, 1.2)
            image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
            image = ensure_contiguous(image)

        # Random rotation (90, 180, 270 degrees)
        if random.random() < 0.3:
            k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
            image = ensure_contiguous(np.rot90(image, k))
            mask = ensure_contiguous(np.rot90(mask, k))

        return image, mask


def collate_fn(batch, batch_size=1):
    """Custom collate function with gradient accumulation support."""
    valid_samples = [sample for sample in batch if sample['valid']]

    if len(valid_samples) == 0:
        return None

    # Ensure all arrays in samples are contiguous
    for sample in valid_samples:
        sample['image'] = ensure_contiguous(sample['image'])
        if len(sample['masks']) > 0:
            sample['masks'] = np.array([ensure_contiguous(mask) for mask in sample['masks']])
        if len(sample['boxes']) > 0:
            sample['boxes'] = ensure_contiguous(sample['boxes'])

    if len(valid_samples) >= batch_size:
        return valid_samples[:batch_size]
    else:
        return valid_samples


def read_images_and_labels(root_path, data_name, type_T):
    """Read images and mask labels, return data list with scientific format support."""
    images_path = os.path.join(root_path, data_name, type_T, 'images')
    masks_path = os.path.join(root_path, data_name, type_T, 'masks')

    data = []

    # Extended list of supported formats
    supported_extensions = ['.jpg', '.png', '.jpeg', '.tif', '.tiff', '.czi', '.nd2']

    for image_name in os.listdir(images_path):
        if any(image_name.lower().endswith(ext) for ext in supported_extensions):
            image_path = os.path.join(images_path, image_name)

            base_name = os.path.splitext(image_name)[0]
            mask_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
            mask_path = None

            for ext in mask_extensions:
                potential_mask = os.path.join(masks_path, base_name + ext)
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break

            if mask_path:
                data.append({"image": image_path, "annotation": mask_path})

    return data


def create_dataloaders(root_path, data_name, type_T='train', batch_size=1, num_workers=0,
                       image_size=512, max_instances=20, train_augment=True, train_size=0.8,
                       use_albumentations=True, suppress_warnings=True,
                       disable_augment_after_epoch=None):
    """
    Create dataloaders for segmentation training/validation with scientific format support.

    Args:
        root_path: Root path of dataset
        data_name: Dataset name
        type_T: Data type ('train', 'val', 'test')
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        image_size: Image size
        max_instances: Max number of instances per image
        train_augment: Whether to use data augmentation for training
        train_size: Training set ratio (0-1)
        use_albumentations: Whether to use albumentations for augmentation
        suppress_warnings: Suppress warning messages
        disable_augment_after_epoch: Disable augmentation after this epoch (None means always on)
    Returns:
        If type_T='train': returns (train_loader, val_loader)
        If type_T='val' or 'test': returns val_loader
    """

    # Set more warning filters if needed
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    # Check albumentations availability
    if use_albumentations and not ALBUMENTATIONS_AVAILABLE:
        if not suppress_warnings:
            print("Warning: albumentations requested but not available. Using basic augmentation.")
        use_albumentations = False

    if type_T == 'train':
        # Read training data
        all_data = read_images_and_labels(root_path, data_name, 'train')

        if train_size == 1.0:
            # Check if val/test folders exist
            val_path = os.path.join(root_path, data_name, 'val')
            test_path = os.path.join(root_path, data_name, 'test')

            if os.path.exists(val_path):
                print("Found 'val' folder, using it as validation set")
                train_data = all_data
                val_data = read_images_and_labels(root_path, data_name, 'val')
            elif os.path.exists(test_path):
                print("Found 'test' folder, using it as validation set")
                train_data = all_data
                val_data = read_images_and_labels(root_path, data_name, 'test')
            else:
                print("No 'val' or 'test' folder found, forcing train_size=0.8")
                train_size = 0.8
                split_idx = int(train_size * len(all_data))
                train_data = all_data[:split_idx]
                val_data = all_data[split_idx:]
        else:
            # Split data by ratio
            split_idx = int(train_size * len(all_data))
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Batch size: {batch_size}")
        print(f"Max instances per image: {max_instances}")
        print(f"Image size: {image_size}")
        print(f"Using albumentations: {use_albumentations}")
        print(f"Scientific format support:")
        print(f"  - TIFF (enhanced): {TIFFFILE_AVAILABLE}")
        print(f"  - CZI: {CZI_AVAILABLE}")
        print(f"  - ND2: {ND2_AVAILABLE}")
        if disable_augment_after_epoch is not None:
            print(f"Data augmentation will be disabled after epoch {disable_augment_after_epoch}")

        # Create datasets
        train_dataset = SegmentationDataset(
            train_data,
            max_size=image_size,
            augment=train_augment,
            max_instances=max_instances,
            use_albumentations=use_albumentations,
            disable_augment_after_epoch=disable_augment_after_epoch
        )
        val_dataset = SegmentationDataset(
            val_data,
            max_size=image_size,
            augment=False,
            max_instances=max_instances,
            use_albumentations=False  # No augmentation for validation
        )

        # Collate functions
        def train_collate_fn(batch):
            return collate_fn(batch, batch_size)

        def val_collate_fn(batch):
            return collate_fn(batch, 1)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=val_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    else:  # type_T in ['val', 'test']
        # Only create validation/test dataloader
        val_data = read_images_and_labels(root_path, data_name, type_T)

        print(f"{type_T.capitalize()} samples: {len(val_data)}")
        print(f"Max instances per image: {max_instances}")
        print(f"Image size: {image_size}")
        print(f"Scientific format support:")
        print(f"  - TIFF (enhanced): {TIFFFILE_AVAILABLE}")
        print(f"  - CZI: {CZI_AVAILABLE}")
        print(f"  - ND2: {ND2_AVAILABLE}")

        val_dataset = SegmentationDataset(
            val_data,
            max_size=image_size,
            augment=False,
            max_instances=max_instances,
            use_albumentations=False  # No augmentation for validation/test
        )

        def val_collate_fn(batch):
            return collate_fn(batch, 1)

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=val_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

        return val_loader


def check_dependencies():
    """Check and display the status of all dependencies."""
    print("=== Dependency Status ===")

    # Core dependencies
    print("Core dependencies:")
    print(f"  ✓ OpenCV: Available")
    print(f"  ✓ NumPy: Available")

    # Augmentation
    print("\nAugmentation:")
    if ALBUMENTATIONS_AVAILABLE:
        print(f"  ✓ Albumentations: v{AL_VERSION}")
    else:
        print("  ✗ Albumentations: Not installed")
        print("    Install with: pip install albumentations")

    # Scientific image formats
    print("\nScientific image formats:")
    if TIFFFILE_AVAILABLE:
        print("  ✓ TiffFile: Available (enhanced TIFF support)")
    else:
        print("  ✗ TiffFile: Not installed (will use OpenCV for TIFF)")
        print("    Install with: pip install tifffile")

    if CZI_AVAILABLE:
        print("  ✓ CZI: Available")
    else:
        print("  ✗ CZI: Not installed")
        print("    Install with: pip install czifile")

    if ND2_AVAILABLE:
        if 'nd2' in globals():
            print("  ✓ ND2: Available (nd2 package)")
        else:
            print("  ✓ ND2: Available (nd2reader package)")
    else:
        print("  ✗ ND2: Not installed")
        print("    Install with: pip install nd2  OR  pip install nd2reader")

    if PIL_AVAILABLE:
        print("  ✓ PIL/Pillow: Available")
    else:
        print("  ✗ PIL/Pillow: Not installed")
        print("    Install with: pip install Pillow")

    print("\n=== Installation Commands ===")
    print("To install all scientific format dependencies:")
    print("pip install tifffile czifile nd2 Pillow albumentations")
    print("\nAlternative ND2 reader:")
    print("pip install nd2reader")


# Usage example:
if __name__ == "__main__":
    # Check dependencies
    check_dependencies()

    print("\n" + "=" * 50)
    print("Testing with enhanced scientific format support")
    print("=" * 50)

    # Training mode - using albumentations augmentation with scientific formats
    try:
        train_loader, val_loader = create_dataloaders(
            root_path="datasets",
            data_name="Intestine",
            type_T='train',
            batch_size=4,
            num_workers=4,
            image_size=512,
            max_instances=20,
            train_augment=True,
            train_size=0.8,
            use_albumentations=True,  # Use albumentations
            disable_augment_after_epoch=10
        )

        print(f"\n✓ Successfully created train_loader with {len(train_loader)} batches")
        print(f"✓ Successfully created val_loader with {len(val_loader)} batches")

        # Test loading a batch
        print("\nTesting batch loading...")
        for i, batch in enumerate(train_loader):
            if batch is not None:
                print(f"✓ Batch {i}: {len(batch)} samples")
                for j, sample in enumerate(batch):
                    img_shape = sample['image'].shape
                    num_masks = len(sample['masks']) if len(sample['masks']) > 0 else 0
                    print(f"  Sample {j}: Image {img_shape}, Masks: {num_masks}")
                break
            if i >= 2:  # Test only first few batches
                break

    except Exception as e:
        print(f"Error creating dataloaders: {e}")

    # Validation mode - testing different scientific formats
    print("\n" + "=" * 30)
    print("Testing individual format loaders")
    print("=" * 30)

    # Test format detection and loading
    test_files = [
        ("test.tif", "TIFF"),
        ("test.tiff", "TIFF"),
        ("test.czi", "CZI"),
        ("test.nd2", "ND2"),
        ("test.jpg", "Standard")
    ]

    for filename, format_type in test_files:
        print(f"\n{format_type} format ({filename}):")
        try:
            # This would normally load a real file
            # Here we just test the format detection logic
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.tif', '.tiff']:
                status = "✓ Supported" if TIFFFILE_AVAILABLE else "⚠ Basic support (OpenCV)"
            elif ext == '.czi':
                status = "✓ Supported" if CZI_AVAILABLE else "✗ Not supported"
            elif ext == '.nd2':
                status = "✓ Supported" if ND2_AVAILABLE else "✗ Not supported"
            else:
                status = "✓ Supported (OpenCV)"

            print(f"  {status}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'=' * 50}")
    print("Enhanced dataset with scientific format support ready!")
    print("Supported formats: TIFF, CZI, ND2, PNG, JPG, JPEG")
    print("=" * 50)