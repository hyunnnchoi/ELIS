"""
ELIS Output Token Predictor - Dataset Module

Loads training data from vllm_results_training.jsonl files across multiple models.
Each sample contains:
  - input_prompt: user prompt (+ already generated partial answer)
  - output_prompt: generated text so far
  - number_of_output_tokens: tokens generated
  - remaining_tokens: tokens left to generate (LABEL)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class ELISDataset(Dataset):
    """Dataset for ELIS output token prediction."""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        max_length: int = 512,
        model_names: List[str] = None
    ):
        """
        Args:
            data_dir: Path to data directory containing model folders
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length for BGE model
            model_names: List of model folder names to include (None = all)
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data from all models
        self.samples = []
        self._load_data(model_names)
        
        print(f"Loaded {len(self.samples)} training samples from {len(model_names or 'all')} models")
    
    def _load_data(self, model_names: List[str] = None):
        """Load all JSONL files from data directory."""
        
        # Get list of model directories
        if model_names is None:
            model_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        else:
            model_dirs = [self.data_dir / name for name in model_names]
        
        # Load data from each model
        for model_dir in tqdm(model_dirs, desc="Loading data from models"):
            jsonl_path = model_dir / "vllm_results_training.jsonl"
            
            if not jsonl_path.exists():
                print(f"Warning: {jsonl_path} not found, skipping")
                continue
            
            # Read JSONL file
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        # Combine input_prompt and output_prompt as the full context
                        # This represents "what the model has seen and generated so far"
                        input_text = data['input_prompt']
                        output_text = data.get('output_prompt', '')
                        
                        # Full context = original prompt + generated text so far
                        full_context = input_text + " " + output_text if output_text else input_text
                        
                        # Label is the remaining tokens to generate
                        remaining_tokens = float(data['remaining_tokens'])
                        
                        self.samples.append({
                            'text': full_context,
                            'label': remaining_tokens,
                            'model': model_dir.name
                        })
                        
                    except Exception as e:
                        print(f"Error parsing line in {jsonl_path}: {e}")
                        continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Tokenize text
        encoded = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample['label'], dtype=torch.float32)
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        labels = [s['label'] for s in self.samples]
        return {
            'num_samples': len(self.samples),
            'mean_remaining_tokens': sum(labels) / len(labels),
            'min_remaining_tokens': min(labels),
            'max_remaining_tokens': max(labels),
            'models': list(set(s['model'] for s in self.samples))
        }


def load_datasets(
    data_dir: str,
    tokenizer_name: str = "BAAI/bge-base-en-v1.5",
    max_length: int = 512,
    seed: int = 42
) -> Tuple[ELISDataset, ELISDataset, ELISDataset]:
    """
    Load and split dataset into train, validation, and test sets.
    Split ratio: 6:2:2 (60% train, 20% val, 20% test)
    
    Args:
        data_dir: Path to data directory
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
        seed: Random seed for reproducibility
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # [NOTE, hyunnnchoi, 2025.11.17] Load full dataset
    full_dataset = ELISDataset(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    
    # [NOTE, hyunnnchoi, 2025.11.17] Split into train/val/test with 6:2:2 ratio
    total_size = len(full_dataset)
    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)
    test_size = total_size - train_size - val_size
    
    # Use torch's random_split for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    print(f"\nDataset split (6:2:2):")
    print(f"  Train: {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({len(test_dataset)/total_size*100:.1f}%)")
    
    # Print statistics
    stats = full_dataset.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Mean remaining tokens: {stats['mean_remaining_tokens']:.2f}")
    print(f"  Min remaining tokens: {stats['min_remaining_tokens']:.2f}")
    print(f"  Max remaining tokens: {stats['max_remaining_tokens']:.2f}")
    print(f"  Models: {', '.join(stats['models'])}")
    
    return train_dataset, val_dataset, test_dataset

