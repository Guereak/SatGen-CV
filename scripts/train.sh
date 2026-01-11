#!/bin/bash

# Training script for Pix2Pix road generation model
# This script will automatically resume from best_model.pth if it exists

set -e  # Exit on error

# Default parameters
EPOCHS=200
BATCH_SIZE=16
LR=0.0002
LAMBDA_L1=100
CHECKPOINT_DIR="checkpoints"
RESUME_CHECKPOINT="${CHECKPOINT_DIR}/best_model.pth"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --lambda-l1)
            LAMBDA_L1="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            RESUME_CHECKPOINT="${CHECKPOINT_DIR}/best_model.pth"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --no-resume)
            RESUME_CHECKPOINT=""
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs EPOCHS              Number of epochs (default: 200)"
            echo "  --batch-size SIZE            Batch size (default: 16)"
            echo "  --lr LR                      Learning rate (default: 0.0002)"
            echo "  --lambda-l1 LAMBDA           L1 loss weight (default: 100)"
            echo "  --checkpoint-dir DIR         Checkpoint directory (default: checkpoints)"
            echo "  --resume CHECKPOINT          Resume from specific checkpoint"
            echo "  --no-resume                  Start training from scratch"
            echo "  --help                       Show this help message"
            echo ""
            echo "By default, training will resume from checkpoints/best_model.pth if it exists."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python -m src.models.train"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --lr $LR"
CMD="$CMD --lambda-l1 $LAMBDA_L1"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"

# Check if we should resume from checkpoint
if [ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
    echo "Resuming training from: $RESUME_CHECKPOINT"
    CMD="$CMD --resume $RESUME_CHECKPOINT"
else
    if [ -n "$RESUME_CHECKPOINT" ]; then
        echo "Checkpoint not found: $RESUME_CHECKPOINT"
    fi
    echo "Starting training from scratch"
fi

# Print configuration
echo "================================"
echo "Training Configuration"
echo "================================"
echo "Epochs:          $EPOCHS"
echo "Batch Size:      $BATCH_SIZE"
echo "Learning Rate:   $LR"
echo "Lambda L1:       $LAMBDA_L1"
echo "Checkpoint Dir:  $CHECKPOINT_DIR"
echo "================================"
echo ""

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""
$CMD
