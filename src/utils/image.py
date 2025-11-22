import numpy as np
from pathlib import Path


def white_pixel_percentage(img):
    if img.ndim == 2:
        # Single channel (grayscale)
        white = img == 255
    else:
        # Multi-channel (RGB/RGBA)
        white = (img == 255).all(-1)
    return 100 * white.mean()


def find_matching_label_file(train_file, labels_dir):
    """
    Find the corresponding label file for a train file, handling extension mismatches.

    Args:
        train_file: Path to the train file
        labels_dir: Path to the labels directory

    Returns:
        Path to the matching label file, or None if not found
    """
    stem = Path(train_file).stem
    labels_dir = Path(labels_dir)
    # Try common image extensions
    extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']

    for ext in extensions:
        label_file = labels_dir / f"{stem}{ext}"
        if label_file.exists():
            return label_file

    return None


def calculate_crop_coverage_stats(metadata, patch_size=256, image_size=(1500, 1500)):
    """
    Calculate coverage statistics for crops from metadata.

    Computes:
    - Percentage of image area covered by at least one crop
    - Percentage of image area covered more than once (overlaps)
    - Average overlap depth (how many crops cover each covered pixel)

    Args:
        metadata: List of metadata dicts (from get_n_random_crops_per_image)
        patch_size: Size of the crop patches
        image_size: Tuple (height, width) of the original images

    Returns:
        Dictionary with overall statistics and per-image statistics
    """
    # Group metadata by source file
    crops_by_file = {}
    for item in metadata:
        source = item['source_file']
        if source not in crops_by_file:
            crops_by_file[source] = []
        crops_by_file[source].append(item)

    all_stats = []

    for filename, crops in crops_by_file.items():
        # Create a coverage map to count how many crops cover each pixel
        h, w = image_size
        coverage_map = np.zeros((h, w), dtype=np.int32)

        # Mark all pixels covered by each crop
        for crop_info in crops:
            x, y = crop_info['x'], crop_info['y']
            coverage_map[y:y + patch_size, x:x + patch_size] += 1

        # Calculate statistics
        total_pixels = h * w
        covered_pixels = np.sum(coverage_map > 0)
        overlap_pixels = np.sum(coverage_map > 1)

        # Percentages
        coverage_pct = 100 * covered_pixels / total_pixels
        overlap_pct = 100 * overlap_pixels / total_pixels

        # Additional metrics
        if covered_pixels > 0:
            overlap_ratio = overlap_pixels / covered_pixels  # ratio of overlapped to covered
            avg_overlap_depth = np.sum(coverage_map) / covered_pixels  # avg crops per covered pixel
        else:
            overlap_ratio = 0
            avg_overlap_depth = 0

        max_overlap_depth = np.max(coverage_map)

        stats = {
            'source_file': filename,
            'num_crops': len(crops),
            'total_area_covered_pct': coverage_pct,
            'overlap_area_pct': overlap_pct,
            'overlap_ratio': overlap_ratio,  # what fraction of covered area is overlapped
            'avg_overlap_depth': avg_overlap_depth,
            'max_overlap_depth': max_overlap_depth,
            'coverage_map': coverage_map
        }

        all_stats.append(stats)

    # Calculate overall statistics across all images
    overall = {
        'num_images': len(all_stats),
        'total_crops': sum(s['num_crops'] for s in all_stats),
        'avg_coverage_pct': np.mean([s['total_area_covered_pct'] for s in all_stats]),
        'avg_overlap_pct': np.mean([s['overlap_area_pct'] for s in all_stats]),
        'avg_overlap_ratio': np.mean([s['overlap_ratio'] for s in all_stats]),
        'avg_overlap_depth': np.mean([s['avg_overlap_depth'] for s in all_stats]),
        'per_image_stats': all_stats
    }

    return overall


