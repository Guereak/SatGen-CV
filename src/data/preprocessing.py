import numpy as np
from pathlib import Path
from PIL import Image
import os, argparse
import yaml

from src.utils.image import find_matching_label_file, white_pixel_percentage


def load_resize_config(config_path=None):
    """
    Load resize configuration from config.yaml

    Args:
        config_path: Path to config.yaml file. If None, tries to find it in project root.

    Returns:
        Dictionary mapping dataset names to target sizes [width, height], or empty dict if not found
    """
    if config_path is None:
        # Try to find config.yaml in common locations
        possible_paths = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    if config_path is None or not Path(config_path).exists():
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config.get('preprocessing', {}).get('resize', {})


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
        # Find corresponding label file (handles extension mismatches)
        label_file = find_matching_label_file(train_file, labels_dir)

        if label_file is None:
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
        
    # print(f"Processed {train_file.name}: {len(train_patches)} patches")
    
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
        label_file = find_matching_label_file(train_file, labels_dir)

        if label_file is None:
            print(f"Warning: No label found for {train_file.name}, skipping...")
            continue

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
        label_file = find_matching_label_file(train_file, labels_dir)

        if label_file is None:
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


def remove_blank_patches(
        input_data_path, output_data_path, train_subdir="train", labels_subdir="train_labels",
        train_max_threshold=5.0, label_min_threshold=2.0
):
    """
    Filter dataset to remove undesired patches

    Args:
        input_data_path: Path to input data directory
        output_data_path: Path to output data directory
        train_subdir: Name of training images subdirectory
        labels_subdir: Name of labels subdirectory
        train_max_threshold: Max percentage of white pixels in train image before discarding
        label_min_threshold: Min percentage of white pixels in label image before discarding
    """

    extensions = ['*.png', '*.tif*', '*.jpg']
    count = 0

    os.makedirs(output_data_path + train_subdir, exist_ok=True)
    os.makedirs(output_data_path + labels_subdir, exist_ok=True)

    train_patches = sorted([f for ext in extensions for f in Path(input_data_path + train_subdir).glob(ext)])
    label_patches = sorted([f for ext in extensions for f in Path(input_data_path + labels_subdir).glob(ext)])

    for train, label in zip(train_patches, label_patches):
        train_img = np.array(Image.open(train))
        label_img = np.array(Image.open(label))

        if (white_pixel_percentage(train_img) < train_max_threshold
                and white_pixel_percentage(label_img) > label_min_threshold):
            count += 1
            base_name = train.stem
            Image.fromarray(train_img).save(Path(output_data_path) / train_subdir / f"{base_name}.png")
            Image.fromarray(label_img).save(Path(output_data_path) / labels_subdir / f"{base_name}.png")

    print(f"Kept: {count / len(train_patches) * 100:.2f}% of patches.")


def process_dataset_split(
    dataset_path,
    split_name,
    output_base_dir,
    dataset_name,
    images_subdir="images",
    labels_subdir="gt",
    patch_size=256,
    overlap=True,
    train_max_threshold=5.0,
    label_min_threshold=2.0,
    resize_target=None,
):
    """
    Process a single split (train/test/val) of a dataset with filtering.
    Extracts patches and applies filtering in one pass.

    Args:
        dataset_path: Path to the dataset split (e.g., data/raw/AerialImageDataset/train)
        split_name: Name of the split (train, test, val)
        output_base_dir: Base output directory (e.g., data/processed)
        dataset_name: Name of the dataset (e.g., AerialImageDataset)
        images_subdir: Subdirectory containing images
        labels_subdir: Subdirectory containing labels
        patch_size: Size of square patches
        overlap: Whether patches should overlap
        train_max_threshold: Max percentage of white pixels in train image
        label_min_threshold: Min percentage of white pixels in label image
        resize_target: Optional tuple/list [width, height] to resize images before cropping

    Returns:
        Number of patches saved
    """
    images_dir = Path(dataset_path) / images_subdir
    labels_dir = Path(dataset_path) / labels_subdir

    if not images_dir.exists() or not labels_dir.exists():
        print(f"  Skipping {split_name}: directories not found")
        return 0

    # Create output directory
    output_dir = Path(output_base_dir) / f"{split_name}_patches_{patch_size}" / dataset_name
    output_images_dir = output_dir / images_subdir
    output_labels_dir = output_dir / labels_subdir
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    extensions = ["*.tif*", "*.png", "*.jpg"]
    image_files = sorted([f for ext in extensions for f in images_dir.glob(ext)])

    total_patches = 0
    saved_patches = 0

    print(f"  Processing {split_name} split: {len(image_files)} images")

    for img_file in image_files:
        label_file = find_matching_label_file(img_file, labels_dir)

        if label_file is None:
            print(f"    Warning: No label found for {img_file.name}, skipping...")
            continue

        # Load images
        img = Image.open(img_file)
        label = Image.open(label_file)

        # Resize if needed
        if resize_target is not None:
            target_width, target_height = resize_target
            original_size = img.size
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            label = label.resize((target_width, target_height), Image.Resampling.NEAREST)
            if img_file == image_files[0]:  # Print only for first image
                print(f"    Resizing images from {original_size} to {(target_width, target_height)}")

        # Extract patches
        img_patches, coords = crop_image_into_patches(img, patch_size, overlap)
        label_patches, _ = crop_image_into_patches(label, patch_size, overlap)

        total_patches += len(img_patches)

        # Filter and save patches
        base_name = img_file.stem
        for i, (img_patch, label_patch, coord) in enumerate(zip(img_patches, label_patches, coords)):
            # Apply filtering
            if (white_pixel_percentage(img_patch) < train_max_threshold and
                white_pixel_percentage(label_patch) > label_min_threshold):

                # Save filtered patch
                patch_name = f"{base_name}_patch_{i:03d}_x{coord[0]}_y{coord[1]}.png"
                Image.fromarray(img_patch).save(output_images_dir / patch_name)
                Image.fromarray(label_patch).save(output_labels_dir / patch_name)
                saved_patches += 1

    keep_percentage = (saved_patches / total_patches * 100) if total_patches > 0 else 0
    print(f"    Kept {saved_patches}/{total_patches} patches ({keep_percentage:.1f}%)")

    return saved_patches


def process_all_datasets(
    raw_data_dir="data/raw",
    output_dir="data/processed",
    patch_size=256,
    overlap=True,
    train_max_threshold=5.0,
    label_min_threshold=2.0,
    images_subdir="images",
    labels_subdir="gt",
    config_path=None,
):
    """
    Process all datasets in the raw data directory.
    Creates organized structure: data/processed/{train,test,val}_patches_256/<dataset_name>/
    """
    raw_data_path = Path(raw_data_dir)

    if not raw_data_path.exists():
        raise ValueError(f"Raw data directory not found: {raw_data_dir}")

    # Load resize configuration
    resize_config = load_resize_config(config_path)
    if resize_config:
        print(f"Loaded resize config: {resize_config}")

    # Find all datasets (subdirectories in raw data dir)
    datasets = [d for d in raw_data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"Found {len(datasets)} datasets: {[d.name for d in datasets]}")
    print(f"Output directory: {output_dir}")
    print(f"Patch size: {patch_size}, Overlap: {overlap}")
    print(f"Filters: train_max_white={train_max_threshold}%, label_min_white={label_min_threshold}%")
    print()

    total_saved = 0

    for dataset_dir in datasets:
        dataset_name = dataset_dir.name
        print(f"Processing dataset: {dataset_name}")

        # Get resize target for this dataset if configured
        resize_target = resize_config.get(dataset_name)
        if resize_target:
            print(f"  Will resize to: {resize_target}")

        # Process each split (train, test, val)
        for split in ["train", "test", "val"]:
            split_path = dataset_dir / split
            if split_path.exists():
                saved = process_dataset_split(
                    dataset_path=split_path,
                    split_name=split,
                    output_base_dir=output_dir,
                    dataset_name=dataset_name,
                    images_subdir=images_subdir,
                    labels_subdir=labels_subdir,
                    patch_size=patch_size,
                    overlap=overlap,
                    train_max_threshold=train_max_threshold,
                    label_min_threshold=label_min_threshold,
                    resize_target=resize_target,
                )
                total_saved += saved

        print()

    print(f"Total patches saved: {total_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and filter patches from dataset")
    parser.add_argument("--raw-data-dir", type=str, default="data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Path to output directory")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Size of square patches")
    parser.add_argument("--images-subdir", type=str, default="images",
                        help="Name of images subdirectory")
    parser.add_argument("--labels-subdir", type=str, default="gt",
                        help="Name of labels subdirectory")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Disable overlapping patches")
    parser.add_argument("--train-max-white", type=float, default=5.0,
                        help="Max percentage of white pixels in train image")
    parser.add_argument("--label-min-white", type=float, default=2.0,
                        help="Min percentage of white pixels in label image")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml file (auto-detected if not specified)")

    args = parser.parse_args()

    process_all_datasets(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap=not args.no_overlap,
        train_max_threshold=args.train_max_white,
        label_min_threshold=args.label_min_white,
        images_subdir=args.images_subdir,
        labels_subdir=args.labels_subdir,
        config_path=args.config,
    )