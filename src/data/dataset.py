import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile
import torchvision.transforms.functional as TF
import random


class SatGenDataset(Dataset):
    """
    Dataset for Pix2Pix road generation.

    Loads paired images:
    - Input: 2-channel segmentation (buildings + roads) from labels
    - Target: 3-channel RGB satellite/aerial imagery from train

    Args:
        images_dir: Directory containing RGB satellite images (.png)
        labels_dir: Directory containing segmentation labels (_processed.tiff)
        augment: Whether to apply data augmentation (default True)
    """
    def __init__(self, images_dir, labels_dir, transform=None, augment=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            input_tensor: [2, H, W] - Buildings and roads segmentation
            target_tensor: [3, H, W] - RGB satellite image
        """
        img_name = self.image_files[idx]
        base_name = img_name.rsplit('.', 1)[0]  # Remove extension
        label_name = f"{base_name}_processed.tiff"

        img_path = self.images_dir / img_name
        label_path = self.labels_dir / label_name

        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        target_img = Image.open(img_path).convert('RGB')
        target_array = np.array(target_img)
        label_array = tifffile.imread(label_path)

        # Check label image is in right format
        assert len(label_array.shape) == 3 and label_array.shape[2] == 2, f"Unexpected label shape: {label_array.shape}"

        input_array = label_array.astype(np.float32)

        # Convert to tensors and normalize
        target_tensor = torch.from_numpy(target_array).permute(2, 0, 1).float() / 255.0
        target_tensor = (target_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

        input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).float()
        if input_tensor.max() > 1.0:
            input_tensor = input_tensor / 255.0
        input_tensor = (input_tensor - 0.5) / 0.5

        # Apply synchronized augmentation transforms
        if self.augment:
            input_tensor, target_tensor = self._apply_augmentation(input_tensor, target_tensor)

        return input_tensor, target_tensor

    def _apply_augmentation(self, input_tensor, target_tensor):
        """
        Apply synchronized random augmentation to both input and target.

        Args:
            input_tensor: [2, H, W] segmentation tensor
            target_tensor: [3, H, W] RGB image tensor

        Returns:
            Augmented input and target tensors
        """
        # Random horizontal flip
        if random.random() > 0.5:
            input_tensor = TF.hflip(input_tensor)
            target_tensor = TF.hflip(target_tensor)

        # Random vertical flip
        if random.random() > 0.5:
            input_tensor = TF.vflip(input_tensor)
            target_tensor = TF.vflip(target_tensor)

        # Random 90-degree rotation (0, 90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            input_tensor = TF.rotate(input_tensor, angle)
            target_tensor = TF.rotate(target_tensor, angle)

        # Random color jitter on target only (brightness, contrast, saturation)
        if random.random() > 0.5:
            brightness_factor = 1.0 + random.uniform(-0.1, 0.1)
            contrast_factor = 1.0 + random.uniform(-0.1, 0.1)
            saturation_factor = 1.0 + random.uniform(-0.1, 0.1)

            # Denormalize, apply jitter, renormalize
            target_tensor = (target_tensor + 1) / 2  # [-1, 1] -> [0, 1]
            target_tensor = TF.adjust_brightness(target_tensor, brightness_factor)
            target_tensor = TF.adjust_contrast(target_tensor, contrast_factor)
            target_tensor = TF.adjust_saturation(target_tensor, saturation_factor)
            target_tensor = torch.clamp(target_tensor, 0, 1)
            target_tensor = (target_tensor - 0.5) / 0.5  # [0, 1] -> [-1, 1]

        return input_tensor, target_tensor


def get_dataloaders(train_root, val_root, batch_size=16, num_workers=4):
    """
    Create train and validation dataloaders.

    Args:
        train_root: Root directory for training data (contains subdirs with images/ and gt/)
        val_root: Root directory for validation data (contains subdirs with images/ and gt/)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, val_loader
    """
    # Collect all images/gt directories from subdirectories
    train_root = Path(train_root)
    val_root = Path(val_root)

    train_images = []
    train_labels = []
    for subdir in train_root.iterdir():
        if subdir.is_dir():
            img_dir = subdir / 'images'
            gt_dir = subdir / 'gt'
            if img_dir.exists() and gt_dir.exists():
                train_images.append(img_dir)
                train_labels.append(gt_dir)

    val_images = []
    val_labels = []
    for subdir in val_root.iterdir():
        if subdir.is_dir():
            img_dir = subdir / 'images'
            gt_dir = subdir / 'gt'
            if img_dir.exists() and gt_dir.exists():
                val_images.append(img_dir)
                val_labels.append(gt_dir)

    # Create combined datasets (augment train, no augment val)
    train_datasets = [SatGenDataset(images_dir=img_dir, labels_dir=gt_dir, augment=True)
                     for img_dir, gt_dir in zip(train_images, train_labels)]
    val_datasets = [SatGenDataset(images_dir=img_dir, labels_dir=gt_dir, augment=False)
                   for img_dir, gt_dir in zip(val_images, val_labels)]

    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
