"""
ELIS Output Token Predictor - Main Training Script

Train BGE-based output token predictor using data from multiple LLM models.

Usage:
    python train.py --data-dir ../data --batch-size 16 --epochs 16
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Import local modules
from dataset import load_datasets
from model import create_model
from trainer import ELISTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ELIS Output Token Predictor"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Path to data directory containing model folders"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for BGE model"
    )
    
    # Model arguments
    parser.add_argument(
        "--bge-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="BGE model name"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1024,
        help="Hidden dimension for FC layers"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of FC layers"
    )
    parser.add_argument(
        "--no-freeze-bge",
        action="store_true",
        help="Do not freeze BGE parameters (fine-tune BGE)"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=16,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval (batches)"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Print configuration
    print("\n" + "="*80)
    print("ELIS Output Token Predictor - Training Configuration")
    print("="*80)
    print(f"\n📁 Data Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Max sequence length: {args.max_length}")
    
    print(f"\n🧠 Model Configuration:")
    print(f"  BGE model: {args.bge_model}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of FC layers: {args.num_layers}")
    print(f"  Freeze BGE: {not args.no_freeze_bge}")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.early_stopping}")
    print(f"  Device: {args.device}")
    print(f"  Random seed: {args.seed}")
    
    print(f"\n💾 Checkpoint Configuration:")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"  Save every: {args.save_every} epoch(s)")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    
    print("\n" + "="*80 + "\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_datasets(
        data_dir=args.data_dir,
        tokenizer_name=args.bge_model,
        max_length=args.max_length,
        seed=args.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    
    print("\n" + "="*80 + "\n")
    
    # Create model
    print("Creating model...")
    model = create_model(
        bge_model_name=args.bge_model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        freeze_bge=not args.no_freeze_bge,
        device=args.device
    )
    
    print("\n" + "="*80 + "\n")
    
    # Create trainer
    trainer = ELISTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resuming from epoch {start_epoch + 1}\n")
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_every=args.save_every
    )
    
    print("\n" + "="*80)
    print("✅ Training completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

