import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Literal, Union
from dataclasses import dataclass
import torch
import tqdm
import tifffile

from .noise_strategies import (
    NoiseStrategy,
    create_noise_strategy,
    is_empty_patch
)


@dataclass
class PatchInfo:
    """Metadata for a generated patch."""
    image: np.ndarray  # (H, W, C) RGB image
    x: int  # Top-left x coordinate in output image
    y: int  # Top-left y coordinate in output image


def create_blend_weights(
    patch_size: int,
    overlap: int,
    mode: Literal["linear", "cosine"] = "cosine"
) -> np.ndarray:
    """ Create a weight matrix for blending patches """
    weights_1d = np.ones(patch_size)
    
    if overlap > 0:
        fade_in = np.linspace(0, 1, overlap, endpoint=False)
        fade_out = np.linspace(1, 0, overlap, endpoint=False)
        
        if mode == "cosine":
            fade_in = 0.5 * (1 - np.cos(np.pi * fade_in))
            fade_out = 0.5 * (1 + np.cos(np.pi * (1 - fade_out)))
        
        weights_1d[:overlap] = fade_in
        weights_1d[-overlap:] = fade_out
    
    return np.outer(weights_1d, weights_1d).astype(np.float32)


def compute_patch_grid(
    output_size: Tuple[int, int],
    patch_size: int,
    overlap: int
) -> List[Tuple[int, int]]:
    """
    Compute the grid of patch positions needed to cover the output image.

    Returns:
        List of (x, y) coordinates for the top-left corner of each patch
    """
    height, width = output_size
    stride = patch_size - overlap
    
    positions = []
    
    y = 0
    while y < height:
        x = 0
        while x < width:
            positions.append((x, y))
            
            # Move to next column
            if x + patch_size >= width:
                break
            x += stride
        
        # Ensure we cover the right edge
        if positions[-1][0] + patch_size < width:
            positions.append((width - patch_size, y))
        
        # Move to next row
        if y + patch_size >= height:
            break
        y += stride
    
    # Ensure we cover the bottom edge
    last_y = positions[-1][1]
    if last_y + patch_size < height:
        # Add a row at the bottom
        x = 0
        while x < width:
            positions.append((x, height - patch_size))
            if x + patch_size >= width:
                break
            x += stride
        if positions[-1][0] + patch_size < width:
            positions.append((width - patch_size, height - patch_size))
    
    return positions


def blend_patches(
    patches: List[PatchInfo],
    output_size: Tuple[int, int],
    overlap: int,
    blend_mode: Literal["linear", "cosine"] = "cosine"
) -> np.ndarray:
    """ Blend multiple overlapping patches into a single seamless image. """
    height, width = output_size
    num_channels = patches[0].image.shape[2]
    patch_size = patches[0].image.shape[0]
    
    output = np.zeros((height, width, num_channels), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    
    weights = create_blend_weights(patch_size, overlap, blend_mode)
    
    # Accumulate weighted patches
    for patch_info in patches:
        x, y = patch_info.x, patch_info.y
        img = patch_info.image.astype(np.float64)
                
        # Handle edge cases where patch extends beyond output
        patch_h, patch_w = img.shape[:2]
        out_x_end = min(x + patch_w, width)
        out_y_end = min(y + patch_h, height)
        patch_x_end = out_x_end - x
        patch_y_end = out_y_end - y
        
        x_start = max(0, x)
        y_start = max(0, y)
        patch_x_start = x_start - x
        patch_y_start = y_start - y
        
        # Extract relevant portions
        patch_region = img[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        weight_region = weights[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        
        # Accumulate
        output[y_start:out_y_end, x_start:out_x_end] += patch_region * weight_region[:, :, np.newaxis]
        weight_sum[y_start:out_y_end, x_start:out_x_end] += weight_region
    
    weight_sum = np.maximum(weight_sum, 1e-8)
    output = output / weight_sum[:, :, np.newaxis]
    
    return output.astype(np.uint8)


class SeamlessGenerator:

    def __init__(
        self,
        model,
        patch_size: int = 256,
        overlap: int = 64,
        blend_mode: Literal["linear", "cosine"] = "cosine",
        device: str = "cpu",
        noise_strategy: Optional[Union[str, NoiseStrategy]] = "gaussian",
        noise_kwargs: Optional[dict] = None,
        empty_threshold: float = 0.01,
        enable_dropout: bool = True
    ):
        """
        Initialize SeamlessGenerator with support for noise injection.

        Args:
            model: PyTorch model (Pix2Pix generator)
            patch_size: Size of patches for processing
            overlap: Overlap between patches
            blend_mode: Blending mode ("linear" or "cosine")
            device: Device to run model on
            noise_strategy: Noise strategy for empty patches
                - String: "gaussian", "sparse", "uniform", "per_channel", "none"
                - NoiseStrategy instance
                - None: No noise
            noise_kwargs: Keyword arguments for noise strategy
            empty_threshold: Threshold for considering a patch "empty"
            enable_dropout: Keep dropout enabled during inference (recommended for Pix2Pix)
        """
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.blend_mode = blend_mode
        self.device = device
        self.empty_threshold = empty_threshold
        self.enable_dropout = enable_dropout

        self.weights = create_blend_weights(patch_size, overlap, blend_mode)

        # Setup noise strategy
        if isinstance(noise_strategy, str):
            noise_kwargs = noise_kwargs or {}
            self.noise_strategy = create_noise_strategy(noise_strategy, **noise_kwargs)
        else:
            self.noise_strategy = noise_strategy

        # Configure model mode for Pix2Pix
        if self.enable_dropout:
            self._enable_pix2pix_dropout()
        else:
            # Standard eval mode (all layers including dropout)
            self.model.eval()

    def _enable_pix2pix_dropout(self):
        """
        Configure model for Pix2Pix inference with dropout enabled.

        In Pix2Pix, dropout is kept enabled during test time to add stochasticity.
        This is different from standard practice where dropout is disabled during inference.

        Configuration:
        - Dropout layers: TRAIN mode (enables dropout during inference)
        - BatchNorm layers: EVAL mode (use running stats, don't update them)
        - Other layers: EVAL mode (no gradient computation)
        """
        # First, set entire model to eval mode
        self.model.eval()

        # Then, explicitly enable dropout layers
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()  # Override: keep dropout in training mode

    def _extract_patch(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        """Extract a patch from the input image, with padding if needed."""
        h, w = image.shape[:2]
        
        x_end = x + self.patch_size
        y_end = y + self.patch_size
        
        # Handle boundary cases with reflection padding
        if x < 0 or y < 0 or x_end > w or y_end > h:
            pad_left = max(0, -x)
            pad_right = max(0, x_end - w)
            pad_top = max(0, -y)
            pad_bottom = max(0, y_end - h)
            
            x_start_src = max(0, x)
            y_start_src = max(0, y)
            x_end_src = min(w, x_end)
            y_end_src = min(h, y_end)
            
            patch = image[y_start_src:y_end_src, x_start_src:x_end_src]
            
            patch = np.pad(
                patch,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='reflect'
            )
        else:
            patch = image[y:y_end, x:x_end]
        
        return patch
    

    def _run_model(self, patch: np.ndarray) -> np.ndarray:
        """
        Run the Pix2Pix generator on a single patch.

        Args:
            patch: Input patch (H, W, C) in range [0, 255]

        Returns:
            Generated patch (H, W, C) in range [0, 255]
        """
        # Check if patch is empty and apply noise if needed
        is_empty = is_empty_patch(patch, threshold=self.empty_threshold)

        if is_empty and self.noise_strategy is not None:
            patch = self.noise_strategy.add_noise(patch, is_empty=True)

        # Preprocess
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        patch_tensor = patch_tensor / 255.0
        patch_tensor = (patch_tensor - 0.5) / 0.5
        patch_tensor = patch_tensor.unsqueeze(0).to(self.device)

        # Run model (note: no_grad is used but dropout is still enabled for Pix2Pix)
        with torch.no_grad():
            output = self.model(patch_tensor)

        # Postprocess
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output + 1) / 2
        output = (output * 255).clip(0, 255).astype(np.uint8)

        return output
    

    def generate(
        self,
        segmentation_mask: np.ndarray,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate a seamless large image from a segmentation mask.

        Args:
            segmentation_mask: Input segmentation mask (H, W, C)
            show_progress: Show progress bar

        Returns:
            Generated RGB image (H, W, 3)
        """
        height, width = segmentation_mask.shape[:2]

        positions = compute_patch_grid(
            (height, width),
            self.patch_size,
            self.overlap
        )

        patches = []
        iterator = tqdm.tqdm(positions, desc="Generating patches") if show_progress else positions

        for x, y in iterator:
            input_patch = self._extract_patch(segmentation_mask, x, y)
            output_patch = self._run_model(input_patch)

            patches.append(PatchInfo(
                image=output_patch,
                x=x,
                y=y
            ))

        if show_progress:
            print("Blending patches...")

        return blend_patches(
            patches,
            (height, width),
            self.overlap,
            self.blend_mode
        )
            

    def generate_from_file(
        self,
        mask_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """ Generate from a segmentation mask file. """        
        mask = tifffile.imread(mask_path)
        
        result = self.generate(mask, show_progress)
        
        if output_path:
            Image.fromarray(result).save(output_path)
        
        return result