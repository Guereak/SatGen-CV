import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile


class SatGenDataset(Dataset):
    """
    Dataset for Pix2Pix road generation.

    Loads paired images:
    - Input: 2-channel segmentation (buildings + roads) from labels
    - Target: 3-channel RGB satellite/aerial imagery from train

    Args:
        images_dir: Directory containing RGB satellite images (.tiff)
        labels_dir: Directory containing segmentation labels (.tif)
    """
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.tiff', '.png'))])

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

        return input_tensor, target_tensor


def get_dataloaders(train_dir, train_labels_dir, val_dir, val_labels_dir,
                    batch_size=16, num_workers=4):
    """
    Create train and validation dataloaders.

    Args:
        train_dir: Path to training images
        train_labels_dir: Path to training labels
        val_dir: Path to validation images
        val_labels_dir: Path to validation labels
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, val_loader
    """
    train_dataset = SatGenDataset(
        images_dir=train_dir,
        labels_dir=train_labels_dir,
    )

    val_dataset = SatGenDataset(
        images_dir=val_dir,
        labels_dir=val_labels_dir,
    )

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
