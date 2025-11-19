import numpy as np
from pathlib import Path
from PIL import Image

def get_n_random_crops_per_image(dataset_path, train_subdir="train", labels_subdir="train_labels",
                                  patch_size=256, n_crops_per_image=5, num_images=-1):
    """
    Extract n random crops from each image in the dataset.
    
    Args:
        dataset_path: Path to dataset directory
        train_subdir: Name of training images subdirectory
        labels_subdir: Name of labels subdirectory
        patch_size: Size of square patches
        n_crops_per_image: Number of random crops to extract per image
        num_images: Number of images to process. If -1 (default), processes all images in the dataset.
                    If positive, processes only the first num_images images.
    
    Returns:
        Dictionary with:
            'train_crops': List of train crop arrays
            'label_crops': List of label crop arrays
            'metadata': List of dicts with source_file, crop_index, x, y coordinates
    """
    train_dir = Path(dataset_path) / train_subdir
    labels_dir = Path(dataset_path) / labels_subdir
    
    if not train_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Directories not found: {train_dir} or {labels_dir}")
    
    train_files = sorted(list(train_dir.glob("*.tif*")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")))
    
    # Limit number of images if specified
    if num_images > 0:
        train_files = train_files[:num_images]
    
    all_train_crops = []
    all_label_crops = []
    metadata = []
    
    for train_file in train_files:
        label_file = labels_dir / train_file.name
        
        if not label_file.exists():
            print(f"Warning: No label found for {train_file.name}, skipping...")
            continue
        
        # Load images
        train_img = np.array(Image.open(train_file))
        label_img = np.array(Image.open(label_file))
        
        h, w = train_img.shape[:2]
        max_y = h - patch_size
        max_x = w - patch_size
        
        # Generate n random crops for this image
        for crop_idx in range(n_crops_per_image):
            x = np.random.randint(0, max_x + 1)
            y = np.random.randint(0, max_y + 1)
            
            # Extract matching crops
            train_crop = train_img[y:y+patch_size, x:x+patch_size]
            label_crop = label_img[y:y+patch_size, x:x+patch_size]
            
            all_train_crops.append(train_crop)
            all_label_crops.append(label_crop)
            
            metadata.append({
                'source_file': train_file.name,
                'crop_index': crop_idx,
                'x': x,
                'y': y
            })
        
        print(f"Extracted {n_crops_per_image} random crops from {train_file.name}")
    
    return {
        'train_crops': all_train_crops,
        'label_crops': all_label_crops,
        'metadata': metadata
    }