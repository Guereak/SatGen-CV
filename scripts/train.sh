#!/bin/bash

# Training script for Pix2Pix road generation model
# This script will automatically resume from best_model.pth if it exists

set -e  # Exit on error

# Default parameters
RESUME_CHECKPOINT="checkpoints/best_model.pth"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""
$CMD
