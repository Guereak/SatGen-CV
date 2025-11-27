import numpy as np
from pathlib import Path

import numpy as np
from PIL import Image
from pathlib import Path
import tifffile


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



def compare_binary_masks(pred_mask, true_mask, threshold=0.5):
    """
    Compare two binary masks (prediction vs ground truth).
    
    Args:
        pred_mask: Predicted mask (PIL Image, numpy array, or file path)
        true_mask: Ground truth mask (PIL Image, numpy array, or file path)
        threshold: Threshold for binarizing predictions (default: 0.5)
    
    Returns:
        dict: Dictionary containing IoU, Precision, Recall, F1-Score, Dice, etc.
    """
    # Load images if they are paths
    if isinstance(pred_mask, (str, Path)):
        pred_mask = tifffile.imread(pred_mask) if str(pred_mask).endswith('.tiff') else np.array(Image.open(pred_mask))
        pred_mask = pred_mask.sum(axis=2).clip(0, 255).astype('uint8')
    elif isinstance(pred_mask, Image.Image):
        pred_mask = np.array(pred_mask)
        pred_mask = pred_mask.sum(axis=2).clip(0, 255).astype('uint8')
    
    if isinstance(true_mask, (str, Path)):
        true_mask = tifffile.imread(true_mask) if str(true_mask).endswith('.tiff') else np.array(Image.open(true_mask))
    elif isinstance(true_mask, Image.Image):
        true_mask = np.array(true_mask)
    
    # Normalize if necessary (values between 0 and 1)
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
    if true_mask.max() > 1:
        true_mask = true_mask / 255.0
    
    # Binarize masks
    pred_binary = (pred_mask > threshold).astype(int).flatten()
    true_binary = (true_mask > threshold).astype(int).flatten()
    
    # Confusion matrix (sans sklearn)
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    
    # Calculate metrics
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Dice coefficient
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'IoU': float(iou),
        'Dice': float(dice),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1),
        'Accuracy': float(accuracy),
        'Specificity': float(specificity),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }

def evaluate_predictions(img2label, threshold=0.5, isAverage=True):
    """
    Evaluate multiple image pairs and return average metrics.
    
    Args:
        img2label: Dictionary mapping {ground_truth_path: prediction_path}
        threshold: Threshold for binarizing predictions (default: 0.5)
        isAverage: boolean to return average data or not
    
    Returns:
        dict: Dictionary with average metrics, standard deviations, and individual results
    """
    results = []
    
    for true_path, pred_path in img2label.items():
        try:
            metrics = compare_binary_masks(pred_path, true_path, threshold)
            metrics['true_path'] = str(true_path.name)
            metrics['pred_path'] = str(pred_path.name)
            results.append(metrics)
        except Exception as e:
            print(f"Error with {true_path.name}: {e}")
            continue
    
    if not results:
        return None
    
    if not isAverage:
        return results
    
    # Calculate averages
    metric_keys = ['IoU', 'Dice', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
    avg_metrics = {key: np.mean([r[key] for r in results]) for key in metric_keys}
    std_metrics = {f'{key}_std': np.std([r[key] for r in results]) for key in metric_keys}
    
    return {
        'average': avg_metrics,
        'std': std_metrics,
        'individual': results,
        'count': len(results)
    }