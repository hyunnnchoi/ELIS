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

# [NOTE, hyunnnchoi, 2025.11.25] A100 80GB 듀얼 GPU용 기본 설정
# [NOTE, hyunnnchoi, 2025.11.25] Per-GPU batch 8 × 2 GPUs = global batch 16 (as per paper Section 4.2)
# Default configuration (per-GPU batch size for DDP)
DATA_DIR="../data"
BATCH_SIZE=8    # per-GPU batch size (global 16 with 2 GPUs, matching paper)
LEARNING_RATE=1e-4
EPOCHS=16
HIDDEN_DIM=1024
NUM_LAYERS=8
CHECKPOINT_DIR="./checkpoints"
NUM_WORKERS=16
NUM_GPUS=2
LOG_TRANSFORM=false  # Set to true if loss is unstable

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
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --log-transform)
            LOG_TRANSFORM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--batch-size N] [--epochs N] [--learning-rate LR] [--num-workers N] [--gpus N] [--log-transform]"
            exit 1
            ;;
    esac
done

GLOBAL_BATCH=$((BATCH_SIZE * NUM_GPUS))

# [NOTE, hyunnnchoi, 2025.11.25] Show per-rank and global configuration
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Per-GPU batch size: $BATCH_SIZE"
echo "  Global batch size: $GLOBAL_BATCH"
echo "  GPUs: $NUM_GPUS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Hidden dimension: $HIDDEN_DIM"
echo "  Number of FC layers: $NUM_LAYERS"
echo "  Number of workers: $NUM_WORKERS"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo "  Log transform labels: $LOG_TRANSFORM"
echo "  Launch mode: torchrun (DDP + FP16)"
echo ""

# [NOTE, hyunnnchoi, 2025.11.25] Build optional flags
EXTRA_FLAGS=""
if [ "$LOG_TRANSFORM" = true ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --log-transform-labels"
fi

# [NOTE, hyunnnchoi, 2025.11.25] Launch training via torchrun for true multi-GPU usage
torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py \
    --data-dir "$DATA_DIR" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-layers "$NUM_LAYERS" \
    --num-workers "$NUM_WORKERS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --distributed \
    --use-mixed-precision \
    --seed 42 \
    $EXTRA_FLAGS

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

