#!/bin/bash

# Preprocessing script for road generation datasets
# This will process all datasets in data/raw/ and organize them into:
#   data/processed/train_patches_256/<dataset_name>/
#   data/processed/test_patches_256/<dataset_name>/

echo "Starting preprocessing pipeline..."
echo "================================="
echo ""

# Run the organized preprocessing
python -m src.data.preprocessing \
    --raw-data-dir data/raw \
    --output-dir data/processed \
    --patch-size 256 \
    --images-subdir images \
    --labels-subdir gt \
    --train-max-white 5.0 \
    --label-min-white 0.0 \
    --test-split-ratio 0.1 \
    --random-seed 42 \

echo ""
echo "================================="
echo "Preprocessing complete!"
echo ""
echo "Output structure:"
echo "  data/processed/train_patches_256/"
echo "  data/processed/test_patches_256/"
echo ""
