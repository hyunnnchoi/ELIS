"""
ELIS Output Token Predictor - Model Module

Architecture:
  1. BGE Model (BAAI/bge-base-en-v1.5) - frozen
  2. Mean pooling over all tokens (including CLS)
  3. 8 fully connected layers with ReLU activation
  4. Output: scalar prediction of remaining tokens
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict


class ELISPredictor(nn.Module):
    """
    ELIS Output Token Predictor.
    
    Uses frozen BGE embeddings + 8 linear layers to predict remaining output tokens.
    """
    
    def __init__(
        self,
        bge_model_name: str = "BAAI/bge-base-en-v1.5",
        hidden_dim: int = 1024,
        num_layers: int = 8,
        freeze_bge: bool = True
    ):
        """
        Args:
            bge_model_name: Pretrained BGE model name
            hidden_dim: Hidden dimension for FC layers
            num_layers: Number of FC layers (default: 8)
            freeze_bge: Whether to freeze BGE parameters (default: True)
        """
        super().__init__()
        
        # [NOTE, hyunnnchoi, 2025.11.17] Load pretrained BGE model
        self.bge_model = AutoModel.from_pretrained(bge_model_name)
        self.bge_hidden_size = self.bge_model.config.hidden_size  # 768 for base model
        
        # Freeze BGE parameters if specified
        if freeze_bge:
            for param in self.bge_model.parameters():
                param.requires_grad = False
            print(f"BGE model parameters frozen: {bge_model_name}")
        
        # [NOTE, hyunnnchoi, 2025.11.17] Build 8 FC layers with ReLU activation
        layers = []
        
        # First layer: BGE output (768) -> hidden_dim (1024)
        layers.append(nn.Linear(self.bge_hidden_size, hidden_dim))
        layers.append(nn.ReLU())
        
        # Middle layers: hidden_dim -> hidden_dim (7 times for total of 8 layers)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Final layer: hidden_dim -> 1 (scalar output)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
        
        print(f"Initialized ELIS Predictor:")
        print(f"  BGE hidden size: {self.bge_hidden_size}")
        print(f"  FC hidden dim: {hidden_dim}")
        print(f"  Number of FC layers: {num_layers}")
        print(f"  Total trainable params: {self.count_trainable_parameters():,}")
    
    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over all tokens (including CLS).
        
        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # Expand attention mask to match embeddings shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings with mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # Sum mask (to get actual token count)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean pooling
        return sum_embeddings / sum_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            predictions: [batch_size, 1] - predicted remaining tokens
        """
        # [NOTE, hyunnnchoi, 2025.11.17] Get BGE embeddings (frozen)
        with torch.no_grad() if not self.bge_model.training else torch.enable_grad():
            outputs = self.bge_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Mean pooling
        pooled_output = self.mean_pooling(token_embeddings, attention_mask)  # [batch_size, hidden_size]
        
        # Pass through FC layers
        predictions = self.predictor(pooled_output)  # [batch_size, 1]
        
        return predictions.squeeze(-1)  # [batch_size]
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_parameters(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    bge_model_name: str = "BAAI/bge-base-en-v1.5",
    hidden_dim: int = 1024,
    num_layers: int = 8,
    freeze_bge: bool = True,
    device: str = "cuda"
) -> ELISPredictor:
    """
    Create and initialize ELIS Predictor model.
    
    Args:
        bge_model_name: Pretrained BGE model name
        hidden_dim: Hidden dimension for FC layers
        num_layers: Number of FC layers
        freeze_bge: Whether to freeze BGE parameters
        device: Device to place model on
    
    Returns:
        model: Initialized ELISPredictor
    """
    model = ELISPredictor(
        bge_model_name=bge_model_name,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        freeze_bge=freeze_bge
    )
    
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing ELIS Predictor model...")
    
    model = create_model(device="cpu")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        predictions = model(dummy_input_ids, dummy_attention_mask)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input_ids.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:3]}")
    
    print("\nModel test passed!")
