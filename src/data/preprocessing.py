import numpy as np
from pathlib import Path
from PIL import Image


def crop_image_into_patches(image, patch_size=256, overlap=False):
    """
    Crop a single image into non-overlapping patches.
    
    Args:
        image: PIL Image or numpy array
        patch_size: Size of square patches (default: 256)
        overlap: If False, patches don't overlap. If True, sliding window with stride=patch_size//2
    
    Returns:
        List of patches as numpy arrays
        List of (x, y) coordinates for each patch
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    h, w = img_array.shape[:2]
    patches = []
    coordinates = []
    
    stride = patch_size if not overlap else patch_size // 2
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_array[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coordinates.append((x, y))
    
    return patches, coordinates


def crop_dataset(dataset_path, train_subdir="train", labels_subdir="train_labels", 
                 patch_size=256, overlap=False, output_dir=None, num_images=-1):
    """
    Crop all images in train and train_labels directories into patches.
    
    Args:
        dataset_path: Path to dataset directory
        train_subdir: Name of training images subdirectory
        labels_subdir: Name of labels subdirectory
        patch_size: Size of square patches
        overlap: Whether patches should overlap
        output_dir: Optional output directory. If None, returns patches in memory
        num_images: Number of images to process. If -1, process all.
        
    Returns:
        Dictionary with 'train_patches', 'label_patches', and 'metadata'
    """
    train_dir = Path(dataset_path) / train_subdir
    labels_dir = Path(dataset_path) / labels_subdir
    
    if not train_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Directories not found: {train_dir} or {labels_dir}")
    
    train_files = sorted(list(train_dir.glob("*.tif*")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")))
    
    all_train_patches = []
    all_label_patches = []
    metadata = []
    
    if num_images != -1:
        train_files = train_files[:num_images]
    
    for train_file in train_files:
        # Find corresponding label file
        label_file = labels_dir / train_file.name
        
        if not label_file.exists():
            print(f"Warning: No label found for {train_file.name}, skipping...")
            continue
        
        # Load images
        train_img = Image.open(train_file)
        label_img = Image.open(label_file)
        
        # Crop into patches (using same coordinates for both)
        train_patches, coords = crop_image_into_patches(train_img, patch_size, overlap)
        label_patches, _ = crop_image_into_patches(label_img, patch_size, overlap)
        
        # Save or store patches
        if output_dir:
            output_path = Path(output_dir)
            output_train = output_path / train_subdir
            output_labels = output_path / labels_subdir
            output_train.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)
            
            base_name = train_file.stem
            for i, (train_patch, label_patch, coord) in enumerate(zip(train_patches, label_patches, coords)):
                patch_name = f"{base_name}_patch_{i:03d}_x{coord[0]}_y{coord[1]}.png"
                Image.fromarray(train_patch).save(output_train / patch_name)
                Image.fromarray(label_patch).save(output_labels / patch_name)
        else:
            all_train_patches.extend(train_patches)
            all_label_patches.extend(label_patches)
            
        # Store metadata
        for i, coord in enumerate(coords):
            metadata.append({
                'source_file': train_file.name,
                'patch_index': i,
                'x': coord[0],
                'y': coord[1]
            })
        
        print(f"Processed {train_file.name}: {len(train_patches)} patches")
    
    return {
        'train_patches': all_train_patches,
        'label_patches': all_label_patches,
        'metadata': metadata
    }


def get_random_crops(dataset_path, train_subdir="train", labels_subdir="train_labels",
                     patch_size=256, num_crops=100):
    """
    Extract random crops from the dataset (useful for quick sampling).
    
    Args:
        dataset_path: Path to dataset directory
        train_subdir: Name of training images subdirectory
        labels_subdir: Name of labels subdirectory
        patch_size: Size of square patches
        num_crops: Number of random crops to extract
    
    Returns:
        train_crops, label_crops (as numpy arrays)
    """
    train_dir = Path(dataset_path) / train_subdir
    labels_dir = Path(dataset_path) / labels_subdir
    
    train_files = sorted(list(train_dir.glob("*.tif*")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")))
    
    train_crops = []
    label_crops = []
    
    for _ in range(num_crops):
        # Pick random file
        train_file = np.random.choice(train_files)
        label_file = labels_dir / train_file.name
        
        # Load images
        train_img = np.array(Image.open(train_file))
        label_img = np.array(Image.open(label_file))
        
        h, w = train_img.shape[:2]
        
        # Random crop position
        max_y = h - patch_size
        max_x = w - patch_size
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)
        
        # Extract crop
        train_crop = train_img[y:y+patch_size, x:x+patch_size]
        label_crop = label_img[y:y+patch_size, x:x+patch_size]
        
        train_crops.append(train_crop)
        label_crops.append(label_crop)
    
    return np.array(train_crops), np.array(label_crops)


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