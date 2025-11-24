"""SAM3 model wrapper for building segmentation."""

import warnings
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Predictor:
    """Wrapper for SAM3 model """

    def __init__(self, device="cpu"):
        """Initialize SAM3 model and processor.

        Args:
            device: Device to run inference on.
        """
        warnings.filterwarnings("ignore")

        device = torch.device(device)
        torch.set_default_device(device)

        self.device = device
        self.model = build_sam3_image_model(device=device)
        self.processor = Sam3Processor(self.model, device=str(device))


    def predict(self, image, prompt):
        """Run raw SAM3 prediction on a single image.

        Args:
            image: Image path or PIL Image object.
            prompt: Text prompt for SAM3 model.

        Returns:
            Tuple of (masks, boxes, scores)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

        return output["masks"], output["boxes"], output["scores"]

    def predict_and_merge(self, image, prompt):
        """Run prediction and return merged mask.

        Merges all predicted masks into a single mask by summing and clipping.

        Args:
            image: Image path or PIL Image object.
            prompt: Text prompt for SAM3 model.
        Returns:
            Merged mask of shape (H, W).
        """
        masks, _, _ = self.predict(image, prompt)

        # Merge masks into one and clip to [0, 1]
        merged_mask = masks.sum(dim=0).squeeze(0).clamp(0, 1)
        merged_mask = merged_mask.cpu().numpy()

        return merged_mask

