#!/bin/bash

# SAM3 Building Segmentation Inference Script
# Runs SAM3 on all datasets in the processed directory

set -e  # Exit on error

# Enable MPS fallback to CPU for unsupported operations (fixes grid_sample bug)
export PYTORCH_ENABLE_MPS_FALLBACK=1

DEVICE="mps"
PROMPT_1="building"
PROMPT_2="road"
PROCESSED_DIR="data/processed"

echo "Starting SAM3 building segmentation inference..."
echo "Device: $DEVICE"
echo "Prompt 1: $PROMPT_1"
echo "Prompt 2: $PROMPT_2"
echo ""

# Process train set - AerialImageDataset
echo "Processing AerialImageDataset (train)..."
python src/models/sam3_model.py \
  --device "$DEVICE" \
  --directory "$PROCESSED_DIR/train_patches_256/AerialImageDataset" \
  --train-subdir images \
  --label-subdir gt \
  --prompt "$PROMPT_2" \
  --output-dir "$PROCESSED_DIR/train_sam3/AerialImageDataset"

echo ""

# Process train set - MassachusettsRoadDataset
echo "Processing MassachusettsRoadDataset (train)..."
python src/models/sam3_model.py \
  --device "$DEVICE" \
  --directory "$PROCESSED_DIR/train_patches_256/MassachusettsRoadDataset" \
  --train-subdir images \
  --label-subdir gt \
  --prompt "$PROMPT_1" \
  --output-dir "$PROCESSED_DIR/train_sam3/MassachusettsRoadDataset"

echo ""


# Process test set - AerialImageDataset
echo "Processing AerialImageDataset (test)..."
python src/models/sam3_model.py \
  --device "$DEVICE" \
  --directory "$PROCESSED_DIR/test_patches_256/AerialImageDataset" \
  --train-subdir images \
  --label-subdir gt \
  --prompt "$PROMPT_2" \
  --output-dir "$PROCESSED_DIR/test_sam3/AerialImageDataset"

echo ""

# Process test set - MassachusettsRoadDataset
echo "Processing MassachusettsRoadDataset (test)..."
python src/models/sam3_model.py \
  --device "$DEVICE" \
  --directory "$PROCESSED_DIR/test_patches_256/MassachusettsRoadDataset" \
  --train-subdir images \
  --label-subdir gt \
  --prompt "$PROMPT_1" \
  --output-dir "$PROCESSED_DIR/test_sam3/MassachusettsRoadDataset"
echo ""

echo "All SAM3 inference completed!"
echo "Results saved in train_sam3/ and test_sam3/ directories."
