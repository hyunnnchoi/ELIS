#!/bin/bash
# ELIS Output Token Predictor - Training Script
# [NOTE, hyunnnchoi, 2025.11.17] Quick start training script

set -e

echo "=========================================="
echo "ELIS Output Token Predictor Training"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: train.py not found. Please run this script from the train/ directory."
    exit 1
fi

# Default configuration (matching paper specifications)
DATA_DIR="../data"
BATCH_SIZE=16
LEARNING_RATE=1e-4
EPOCHS=16
HIDDEN_DIM=1024
NUM_LAYERS=8
CHECKPOINT_DIR="./checkpoints"

# Parse command line arguments (optional)
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--batch-size N] [--epochs N] [--learning-rate LR]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Hidden dimension: $HIDDEN_DIM"
echo "  Number of FC layers: $NUM_LAYERS"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Run training
python train.py \
    --data-dir "$DATA_DIR" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-layers "$NUM_LAYERS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --seed 42

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

