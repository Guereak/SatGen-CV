"""roadgen-cv: Computer vision tools for road generation."""

from src.data.preprocessing import (
    crop_dataset,
    crop_image_into_patches,
    get_n_random_crops_per_image,
    get_random_crops,
    remove_blank_patches,
    run_pipeline,
)
from src.models.sam3_model import SAM3Predictor
from src.utils.image import (
    calculate_crop_coverage_stats,
    find_matching_label_file,
    white_pixel_percentage,
)
from src.utils.visualization import (
    display_masks_with_scores,
    show_crop_sets,
    visualize_coverage_map,
    visualize_crop_locations,
)

__all__ = [
    # Data preprocessing
    "crop_dataset",
    "crop_image_into_patches",
    "get_n_random_crops_per_image",
    "get_random_crops",
    "remove_blank_patches",
    "run_pipeline",
    # Models
    "SAM3Predictor",
    # Inference
    "BatchProcessor",
    "process_patches",
    # Utils
    "calculate_crop_coverage_stats",
    "find_matching_label_file",
    "white_pixel_percentage",
    # Visualization
    "display_masks_with_scores",
    "show_crop_sets",
    "visualize_coverage_map",
    "visualize_crop_locations",
]
