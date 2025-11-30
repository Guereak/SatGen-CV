import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


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

        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.tiff')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            input_tensor: [2, H, W] - Buildings and roads segmentation
            target_tensor: [3, H, W] - RGB satellite image
        """
        img_name = self.image_files[idx]
        label_name = img_name.replace('.tiff', '.tif')

        img_path = self.images_dir / img_name
        label_path = self.labels_dir / label_name

        target_img = Image.open(img_path).convert('RGB')
        target_array = np.array(target_img)

        label_img = Image.open(label_path)
        label_array = np.array(label_img)

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


