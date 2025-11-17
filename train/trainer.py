"""
ELIS Output Token Predictor - Trainer Module

Implements training loop with:
  - MSE loss
  - Adam optimizer
  - MAE and RMSE metrics
  - Checkpoint saving
  - Early stopping
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ELISTrainer:
    """Trainer for ELIS Output Token Predictor."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100
    ):
        """
        Args:
            model: ELIS Predictor model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            learning_rate: Learning rate for optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging interval (steps)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.log_interval = log_interval
        
        # [NOTE, hyunnnchoi, 2025.11.17] Loss function: MSE for regression
        self.criterion = nn.MSELoss()
        
        # [NOTE, hyunnnchoi, 2025.11.17] Optimizer: Adam with lr=1e-4
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'train_rmse': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        if test_loader:
            print(f"  Test batches: {len(test_loader)}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask)
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_labels)))
        rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_labels)) ** 2))
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse
        }
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, split: str = "Val") -> Dict[str, float]:
        """
        Evaluate on validation or test set.
        
        Args:
            data_loader: Data loader to evaluate on
            split: Split name for logging (Val/Test)
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(data_loader, desc=f"[{split}]")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask)
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Accumulate metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(data_loader)
        mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_labels)))
        rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_labels)) ** 2))
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (epoch {epoch}, val_loss={metrics['loss']:.4f})")
        
        # Save latest model
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(
        self,
        num_epochs: int = 16,
        early_stopping_patience: int = 5,
        save_every: int = 1
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_every: Save checkpoint every N epochs
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, split="Val")
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}, RMSE: {train_metrics['rmse']:.2f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered (patience={early_stopping_patience})")
                print(f"Best model was at epoch {self.best_epoch} with val_loss={self.best_val_loss:.4f}")
                break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best model: epoch {self.best_epoch}, val_loss={self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        # Test evaluation if test_loader is provided
        if self.test_loader:
            print("\n" + "="*60)
            print("Evaluating on test set with best model...")
            print("="*60)
            
            # Load best model
            best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
            self.load_checkpoint(str(best_checkpoint_path))
            
            # Evaluate on test set
            test_metrics = self.evaluate(self.test_loader, split="Test")
            
            print(f"\nTest Set Results:")
            print(f"  Loss: {test_metrics['loss']:.4f}")
            print(f"  MAE:  {test_metrics['mae']:.2f} tokens")
            print(f"  RMSE: {test_metrics['rmse']:.2f} tokens")
            
            # Save test results
            test_results = {
                'test_loss': test_metrics['loss'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'best_epoch': self.best_epoch,
                'best_val_loss': self.best_val_loss
            }
            
            test_results_path = self.checkpoint_dir / "test_results.json"
            with open(test_results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"\nTest results saved to {test_results_path}")

