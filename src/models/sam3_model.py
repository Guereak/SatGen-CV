"""SAM3 model wrapper for building segmentation."""

import warnings
from pathlib import Path
import os

import tifffile
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
from dotenv import load_dotenv

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class SAM3Predictor:
    """Wrapper for SAM3 model """

    def __init__(self, device="cpu"):
        """Initialize SAM3 model and processor.

        Args:
            device: Device to run inference on.
        """
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


    def predict_and_process_single(
        self,
        image_path,
        prompt,
        label_img_path,
        output_img_path,
    ):
        """Run prediction on a single image."""
        mask = self.predict_and_merge(image_path, prompt)
        label_img = np.array(Image.open(label_img_path))

        if label_img.ndim == 3:
            label_img = label_img.max(axis=-1)

        stacked = np.stack([
            (mask * 255).astype('uint8'),
            label_img.astype('uint8')
        ], axis=-1)

        if output_img_path:
            tifffile.imwrite(
                output_img_path,
                stacked,
                photometric="minisblack",
                metadata={"axes": "YXC"}
            )

        return stacked

    def predict_and_process_directory(
        self,
        directory_path,
        train_subdir,
        label_subdir,
        prompt,
        max_instances=None,
        output_dir=None
    ):
        """Run prediction on a batch of images and save masks to disk.

        Args:
            directory_path: Root directory containing subdirectories.
            train_subdir: Subdirectory name containing input images.
            label_subdir: Subdirectory name containing label images.
            prompt: Text prompt for SAM3 model.
            max_instances: Maximum number of instances to process. If None, process all.
            output_dir: Directory to save output files. If None, defaults to '../sam3_predictions' relative to directory_path.
        """

        extensions = ['*.png', '*.tif*', '*.jpg']

        if isinstance(directory_path, str):
            directory_path = Path(directory_path)

        train_dir = Path(directory_path) / train_subdir
        labels_dir = Path(directory_path) / label_subdir

        if output_dir is None:
            output_dir = (directory_path / "../sam3_predictions").resolve()
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        train_patches = sorted([f for ext in extensions for f in train_dir.glob(ext)])
        label_patches = sorted([f for ext in extensions for f in labels_dir.glob(ext)])

        iterator = zip(train_patches, label_patches)

        # Limit to max_instances if specified
        if max_instances is not None:
            iterator = list(iterator)[:max_instances]
        else:
            iterator = list(iterator)

        iterator = tqdm(
            iterator,
            desc="Processing patches",
            unit="patch"
        )


        for train_path, label_path in iterator:
            output_file = output_dir / f"{train_path.stem}_processed.tiff"
            if output_file.exists():
                continue

            try:
                self.predict_and_process_single(
                    train_path,
                    prompt,
                    label_path,
                    output_file
                )
            except:
                logger.exception(f"Error processing {train_path}")

        print("All set processed!")


if __name__ == "__main__":
    import argparse

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='cuda')
        parser.add_argument('--directory', default='../../data/processed/filtered_patches_256/')
        parser.add_argument('--train-subdir', default='train')
        parser.add_argument('--label-subdir', default='train_labels')
        parser.add_argument('--prompt', default='building')
        parser.add_argument('--output-dir', default=None, help='Output directory for processed files')

        args = parser.parse_args()

        processor = SAM3Predictor(
            device=args.device,
        )

        processor.predict_and_process_directory(
            directory_path=args.directory,
            train_subdir=args.train_subdir,
            label_subdir=args.label_subdir,
            prompt=args.prompt,
            output_dir=args.output_dir
        )