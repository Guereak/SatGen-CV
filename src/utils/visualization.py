import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.image import find_matching_label_file


def show_crop_sets(train_crops, label_crops, images_per_row=8, num_sets=2, cmap='gray'):
    pairs = [(train_crops, "Train"), (label_crops, "Label")]
    rows = num_sets * 2

    fig, axes = plt.subplots(rows, images_per_row, figsize=(22, 12))

    for r, (crops, title) in enumerate(pairs * num_sets):
        offset = (r // 2) * images_per_row
        for c in range(images_per_row):
            idx = offset + c
            ax = axes[r, c]
            ax.imshow(crops[idx], cmap=cmap)
            ax.set_title(f"{title} {idx+1}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_crop_locations(dataset_path, train_subdir="train", labels_subdir="train_labels",
                             metadata=None, image_name=None, patch_size=256,
                             max_images=5, figsize=(15, 10)):
    """
    Visualize where crops were taken from the original images by drawing bounding boxes.
    
    Args:
        dataset_path: Path to dataset directory
        train_subdir: Name of training images subdirectory
        labels_subdir: Name of labels subdirectory
        metadata: List of metadata dicts (from get_n_random_crops_per_image or crop_dataset)
        image_name: Optional - visualize only this specific image
        patch_size: Size of the crop patches
        max_images: Maximum number of images to visualize (if image_name not specified)
        figsize: Figure size for the plot
    """
    if metadata is None:
        raise ValueError("metadata parameter is required")
    
    train_dir = Path(dataset_path) / train_subdir
    labels_dir = Path(dataset_path) / labels_subdir
    
    # Group metadata by source file
    crops_by_file = {}
    for item in metadata:
        source = item['source_file']
        if source not in crops_by_file:
            crops_by_file[source] = []
        crops_by_file[source].append(item)
    
    # Filter to specific image if requested
    if image_name:
        if image_name not in crops_by_file:
            raise ValueError(f"Image {image_name} not found in metadata")
        crops_by_file = {image_name: crops_by_file[image_name]}
    else:
        # Limit to max_images
        crops_by_file = dict(list(crops_by_file.items())[:max_images])
    
    num_images = len(crops_by_file)
    fig, axes = plt.subplots(num_images, 2, figsize=figsize)
    
    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (filename, crops) in enumerate(crops_by_file.items()):
        # Load the original images
        train_file = train_dir / filename
        train_img = np.array(Image.open(train_file))
        label_file = find_matching_label_file(train_file, labels_dir)
        label_img = np.array(Image.open(label_file))
        
        # Plot train image with bounding boxes
        axes[idx, 0].imshow(train_img)
        axes[idx, 0].set_title(f"Train: {filename}\n({len(crops)} crops)", fontsize=10)
        axes[idx, 0].axis('off')
        
        # Plot label image with bounding boxes
        axes[idx, 1].imshow(label_img)
        axes[idx, 1].set_title(f"Label: {filename}\n({len(crops)} crops)", fontsize=10)
        axes[idx, 1].axis('off')
        
        # Draw bounding boxes for each crop
        colors = plt.cm.rainbow(np.linspace(0, 1, len(crops)))
        
        for crop_idx, crop_info in enumerate(crops):
            x, y = crop_info['x'], crop_info['y']
            color = colors[crop_idx]
            
            # Draw rectangle on train image
            rect_train = mpatches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            axes[idx, 0].add_patch(rect_train)

            
            # Draw rectangle on label image
            rect_label = mpatches.Rectangle(
                (x, y), patch_size, patch_size,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            axes[idx, 1].add_patch(rect_label)
            
    plt.tight_layout()
    plt.show()
    
    return fig


def visualize_coverage_map(coverage_stats, image_idx=0, figsize=(12, 5)):
    """
    Visualize the coverage map showing how many times each pixel is covered.

    Args:
        coverage_stats: Output from calculate_crop_coverage_stats
        image_idx: Index of the image to visualize (default: 0)
        figsize: Figure size
    """
    stats = coverage_stats['per_image_stats'][image_idx]
    coverage_map = stats['coverage_map']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Coverage heatmap
    im = axes[0].imshow(coverage_map, cmap='hot', interpolation='nearest')
    axes[0].set_title(f"Coverage Heatmap\n{stats['source_file']}")
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], label='Number of overlapping crops')

    # Binary coverage (covered vs not covered)
    binary_coverage = (coverage_map > 0).astype(float)
    axes[1].imshow(binary_coverage, cmap='gray', interpolation='nearest')
    axes[1].set_title(f"Binary Coverage\n{stats['total_area_covered_pct']:.1f}% covered")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\nStatistics for {stats['source_file']}:")
    print(f"  Number of crops: {stats['num_crops']}")
    print(f"  Total area covered: {stats['total_area_covered_pct']:.2f}%")
    print(f"  Area with overlaps: {stats['overlap_area_pct']:.2f}%")
    print(
        f"  Overlap ratio: {stats['overlap_ratio']:.2f} ({100 * stats['overlap_ratio']:.1f}% of covered area is overlapped)")
    print(f"  Average overlap depth: {stats['avg_overlap_depth']:.2f} crops per covered pixel")
    print(f"  Max overlap depth: {stats['max_overlap_depth']} crops on same pixel")

    return fig