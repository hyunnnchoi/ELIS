import os
import json
import numpy as np
from datasets import load_dataset
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# [NOTE, hyunnnchoi, 2025.11.11] Configuration for dataset processing
DATASET_NAME = "lmsys/lmsys-chat-1m"
SAMPLE_SIZE = 11000
RANDOM_SEED = 42  # Fixed seed for reproducibility
OUTPUT_FILE = "processed_dataset.json"
DATA_DIR = "./data"


def ensure_data_directory():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"Data directory ensured at: {DATA_DIR}")


def load_or_process_dataset() -> List[str]:
    """
    Load processed dataset from file if exists, otherwise process and save.
    Returns list of user prompts ready for vLLM input.
    """
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    
    # If processed file exists, load it
    if os.path.exists(output_path):
        logger.info(f"Loading existing processed dataset from {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data['prompts'])} prompts from cache")
        return data['prompts']
    
    # Otherwise, process the dataset
    logger.info("Processing dataset from scratch...")
    prompts = process_dataset()
    
    # Save processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'prompts': prompts, 'count': len(prompts)}, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(prompts)} prompts to {output_path}")
    
    return prompts


def process_dataset() -> List[str]:
    """
    Process the full dataset: sample, extract user inputs, remove duplicates and outliers.
    """
    # Step 1: Load and sample dataset
    logger.info(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    
    # Get train split and sample with fixed seed
    train_data = ds['train']
    total_size = len(train_data)
    logger.info(f"Total dataset size: {total_size}")
    
    # Sample with fixed seed for reproducibility
    sampled_ds = train_data.shuffle(seed=RANDOM_SEED).select(range(min(SAMPLE_SIZE, total_size)))
    logger.info(f"Sampled {len(sampled_ds)} examples")
    
    # Step 2: Extract user inputs only
    user_prompts = extract_user_inputs(sampled_ds)
    logger.info(f"Extracted {len(user_prompts)} user prompts")
    
    # Step 3: Remove duplicates
    unique_prompts = remove_duplicates(user_prompts)
    logger.info(f"After removing duplicates: {len(unique_prompts)} prompts")
    
    # Step 4: Remove outliers
    filtered_prompts = remove_outliers(unique_prompts)
    logger.info(f"After removing outliers: {len(filtered_prompts)} prompts")
    
    return filtered_prompts


def extract_user_inputs(dataset) -> List[str]:
    """
    Extract only user inputs from conversation data.
    Handles multi-turn conversations by extracting only user messages.
    """
    user_prompts = []
    
    for example in dataset:
        # The conversation field contains the dialogue
        conversation = example.get('conversation', [])
        
        # Extract all user messages (role == 'user')
        for turn in conversation:
            if isinstance(turn, dict) and turn.get('role') == 'user':
                content = turn.get('content', '').strip()
                if content:  # Only add non-empty prompts
                    user_prompts.append(content)
    
    return user_prompts


def remove_duplicates(prompts: List[str]) -> List[str]:
    """
    Remove duplicate prompts while preserving order.
    """
    seen = set()
    unique_prompts = []
    
    for prompt in prompts:
        # Normalize for comparison (lowercase, strip whitespace)
        normalized = prompt.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_prompts.append(prompt)
    
    return unique_prompts


def remove_outliers(prompts: List[str]) -> List[str]:
    """
    Remove outliers based on text length using IQR method with log transformation.
    """
    if len(prompts) == 0:
        return prompts
    
    # Calculate lengths
    lengths = np.array([len(prompt) for prompt in prompts])
    
    # Apply log transformation (add 1 to avoid log(0))
    log_lengths = np.log1p(lengths)
    
    # Calculate IQR
    q1 = np.percentile(log_lengths, 25)
    q3 = np.percentile(log_lengths, 75)
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter outliers
    filtered_prompts = []
    for prompt, log_len in zip(prompts, log_lengths):
        if lower_bound <= log_len <= upper_bound:
            filtered_prompts.append(prompt)
    
    removed_count = len(prompts) - len(filtered_prompts)
    logger.info(f"Removed {removed_count} outliers (IQR method with log transformation)")
    logger.info(f"Length stats - Min: {lengths.min()}, Max: {lengths.max()}, Mean: {lengths.mean():.2f}")
    
    return filtered_prompts


def get_prompts() -> List[str]:
    """
    Main function to get processed prompts.
    This ensures the data directory exists and returns cached or newly processed prompts.
    """
    ensure_data_directory()
    return load_or_process_dataset()


if __name__ == "__main__":
    # Test the dataset processing
    prompts = get_prompts()
    
    print(f"\n{'='*80}")
    print(f"Total prompts ready for vLLM: {len(prompts)}")
    print(f"{'='*80}")
    print("\nFirst 3 examples:")
    for i, prompt in enumerate(prompts[:3], 1):
        print(f"\n[Example {i}]")
        print(f"Length: {len(prompt)} characters")
        print(f"Content: {prompt[:200]}..." if len(prompt) > 200 else f"Content: {prompt}")
    print(f"\n{'='*80}")
