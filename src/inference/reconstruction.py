"""Utilities for reconstructing images from pre-generated patches."""

import numpy as np
from pathlib import Path
import re
from typing import Literal
import tifffile

from .blending import blend_patches, PatchInfo


def reconstruct_image(
    patches_dir: str,
    image_id: str,
    patch_size: int = 256,
    overlap: int = 128,
    blend_mode: Literal["linear", "cosine"] = "cosine"
) -> np.ndarray:
    """ Reconstruct an image from patches """
    patches_path = Path(patches_dir)

    # Find all patches for this image ID
    pattern = re.compile(rf"^{re.escape(image_id)}_15_patch_\d+_x(\d+)_y(\d+)_processed\.tiff$")
    patch_files = []
    coords = []

    for patch_file in patches_path.glob("*.tiff"):
        match = pattern.match(patch_file.name)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            patch_files.append(patch_file)
            coords.append((x, y))

    if len(patch_files) == 0:
        raise ValueError(f"No patches found for image ID: {image_id}")

    # Sort patches by y then x coordinates
    sorted_indices = sorted(range(len(coords)), key=lambda i: (coords[i][1], coords[i][0]))
    patch_files = [patch_files[i] for i in sorted_indices]
    coords = [coords[i] for i in sorted_indices]

    max_x = max(x for x, y in coords) + patch_size
    max_y = max(y for x, y in coords) + patch_size
    output_size = (max_y, max_x)

    patch_infos = []
    for patch_file, (x, y) in zip(patch_files, coords):
        patch_img = tifffile.imread(patch_file)
        patch_infos.append(PatchInfo(image=patch_img, x=x, y=y))

    reconstructed = blend_patches(patch_infos, output_size, overlap, blend_mode)

    return reconstructed
