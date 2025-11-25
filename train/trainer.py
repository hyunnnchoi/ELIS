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
# [NOTE, hyunnnchoi, 2025.11.25] Add distributed communication utilities
import torch.distributed as dist
# [NOTE, hyunnnchoi, 2025.11.25] Sampler typing for distributed loaders
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm


class ELISTrainer:
    """Trainer for ELIS Output Token Predictor."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        train_sampler: Optional[Sampler] = None,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
        use_mixed_precision: bool = False,
        distributed: bool = False,
        global_rank: int = 0,
        world_size: int = 1
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
        # [NOTE, hyunnnchoi, 2025.11.25] Trainer now tracks sampler and distributed context
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_sampler = train_sampler
        self.device = device
        self.log_interval = log_interval
        self.use_mixed_precision = use_mixed_precision and device.startswith("cuda")
        self.distributed = distributed
        self.world_size = world_size
        self.rank = global_rank
        self.is_master = global_rank == 0
        
        # [NOTE, hyunnnchoi, 2025.11.17] Loss function: MSE for regression
        self.criterion = nn.MSELoss()
        
        # [NOTE, hyunnnchoi, 2025.11.17] Optimizer: Adam with lr=1e-4
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        # [NOTE, hyunnnchoi, 2025.11.25] Mixed precision scaler depends on CUDA availability
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # [NOTE, hyunnnchoi, 2025.11.25] Training history now includes R² as per paper Section 4.2
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'train_rmse': [],
            'train_r2': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        if self.is_master:
            print(f"Trainer initialized:")
            print(f"  Device: {device}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Mixed precision: {self.use_mixed_precision}")
            print(f"  Distributed: {self.distributed} (world_size={self.world_size})")
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
        
        if self.distributed and self.train_sampler is not None:
            # [NOTE, hyunnnchoi, 2025.11.25] Shuffle shards differently each epoch when distributed
            self.train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        abs_error_sum = 0.0
        squared_error_sum = 0.0
        label_sum = 0.0
        label_squared_sum = 0.0
        sample_count = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            disable=not self.is_master
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            batch_size = labels.size(0)
            
            # [NOTE, hyunnnchoi, 2025.01.27] Mixed precision training (FP16)
            if self.use_mixed_precision and self.scaler is not None:
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    predictions = self.model(input_ids, attention_mask)
                    loss = self.criterion(predictions, labels)
                
                # Backward pass with scaler
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                predictions = self.model(input_ids, attention_mask)
                loss = self.criterion(predictions, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # [NOTE, hyunnnchoi, 2025.11.25] Accumulate sums for global-reduced metrics
            detached_preds = predictions.detach()
            detached_labels = labels.detach()
            errors = detached_preds - detached_labels
            
            total_loss += loss.item() * batch_size
            abs_error_sum += torch.abs(errors).sum().item()
            squared_error_sum += torch.pow(errors, 2).sum().item()
            # [NOTE, hyunnnchoi, 2025.11.25] Accumulate label sums for R² calculation
            label_sum += detached_labels.sum().item()
            label_squared_sum += torch.pow(detached_labels, 2).sum().item()
            sample_count += batch_size
            
            # Update progress bar
            if self.is_master and (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # [NOTE, hyunnnchoi, 2025.11.25] All-reduce metrics across workers
        metrics_tensor = torch.tensor(
            [total_loss, abs_error_sum, squared_error_sum, label_sum, label_squared_sum, sample_count],
            device=self.device
        )
        if self.distributed:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        total_loss, abs_error_sum, squared_error_sum, label_sum, label_squared_sum, sample_count = metrics_tensor.tolist()
        avg_loss = total_loss / sample_count if sample_count else 0.0
        mae = abs_error_sum / sample_count if sample_count else 0.0
        rmse = np.sqrt(squared_error_sum / sample_count) if sample_count else 0.0
        
        # [NOTE, hyunnnchoi, 2025.11.25] Calculate R² (coefficient of determination) as per paper Section 4.2
        # R² = 1 - SS_res / SS_tot, where SS_tot = sum((y - y_mean)^2) = sum(y^2) - n * y_mean^2
        if sample_count > 0:
            label_mean = label_sum / sample_count
            ss_tot = label_squared_sum - sample_count * (label_mean ** 2)
            ss_res = squared_error_sum
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r2 = 0.0
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
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
        abs_error_sum = 0.0
        squared_error_sum = 0.0
        label_sum = 0.0
        label_squared_sum = 0.0
        sample_count = 0
        
        progress_bar = tqdm(
            data_loader,
            desc=f"[{split}]",
            disable=not self.is_master
        )
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            batch_size = labels.size(0)
            
            # [NOTE, hyunnnchoi, 2025.01.27] Mixed precision for evaluation (faster)
            if self.use_mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.model(input_ids, attention_mask)
                    loss = self.criterion(predictions, labels)
            else:
                predictions = self.model(input_ids, attention_mask)
                loss = self.criterion(predictions, labels)
            
            detached_labels = labels.detach()
            errors = predictions.detach() - detached_labels
            total_loss += loss.item() * batch_size
            abs_error_sum += torch.abs(errors).sum().item()
            squared_error_sum += torch.pow(errors, 2).sum().item()
            # [NOTE, hyunnnchoi, 2025.11.25] Accumulate label sums for R² calculation
            label_sum += detached_labels.sum().item()
            label_squared_sum += torch.pow(detached_labels, 2).sum().item()
            sample_count += batch_size
        
        metrics_tensor = torch.tensor(
            [total_loss, abs_error_sum, squared_error_sum, label_sum, label_squared_sum, sample_count],
            device=self.device
        )
        if self.distributed:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        total_loss, abs_error_sum, squared_error_sum, label_sum, label_squared_sum, sample_count = metrics_tensor.tolist()
        avg_loss = total_loss / sample_count if sample_count else 0.0
        mae = abs_error_sum / sample_count if sample_count else 0.0
        rmse = np.sqrt(squared_error_sum / sample_count) if sample_count else 0.0
        
        # [NOTE, hyunnnchoi, 2025.11.25] Calculate R² (coefficient of determination) as per paper Section 4.2
        if sample_count > 0:
            label_mean = label_sum / sample_count
            ss_tot = label_squared_sum - sample_count * (label_mean ** 2)
            ss_res = squared_error_sum
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r2 = 0.0
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        if self.distributed and not self.is_master:
            return
        
        # [NOTE, hyunnnchoi, 2025.01.27] Handle DataParallel model state dict
        model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            if self.is_master:
                print(f"✓ Saved best model (epoch {epoch}, val_loss={metrics['loss']:.4f})")
        
        # Save latest model
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # [NOTE, hyunnnchoi, 2025.01.27] Handle DataParallel model state dict
        state_dict = checkpoint['model_state_dict']
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        # Load scaler state if available
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
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
        if self.is_master:
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
            self.history['train_r2'].append(train_metrics['r2'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_r2'].append(val_metrics['r2'])
            
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            if self.is_master:
                print(f"\nEpoch {epoch}/{num_epochs} Summary:")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}, RMSE: {train_metrics['rmse']:.2f}, R²: {train_metrics['r2']:.3f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}, R²: {val_metrics['r2']:.3f}")
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
            if self.is_master and (epoch % save_every == 0 or is_best):
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if self.is_master:
                    print(f"\n⚠ Early stopping triggered (patience={early_stopping_patience})")
                    print(f"Best model was at epoch {self.best_epoch} with val_loss={self.best_val_loss:.4f}")
                break
        
        if self.is_master:
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Best model: epoch {self.best_epoch}, val_loss={self.best_val_loss:.4f}")
            print(f"{'='*60}\n")
        
        # Save training history
        if self.is_master:
            history_path = self.checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"Training history saved to {history_path}")
        
        # Test evaluation if test_loader is provided
        if self.is_master and self.test_loader:
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
            print(f"  R²:   {test_metrics['r2']:.3f}")
            
            # Save test results
            test_results = {
                'test_loss': test_metrics['loss'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_r2': test_metrics['r2'],
                'best_epoch': self.best_epoch,
                'best_val_loss': self.best_val_loss
            }
            
            test_results_path = self.checkpoint_dir / "test_results.json"
            with open(test_results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"\nTest results saved to {test_results_path}")

