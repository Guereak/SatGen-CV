import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        train_img = np.array(Image.open(train_dir / filename))
        label_img = np.array(Image.open(labels_dir / filename))
        
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

