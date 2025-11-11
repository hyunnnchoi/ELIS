# ELIS Dataset Processor

A preprocessing tool for the lmsys-chat-1m dataset, preparing user prompts for vLLM input.

## Features

1. **Deterministic Sampling**: Uses a fixed random seed (42) to ensure consistent 11,000 sample selection across runs
2. **User Input Extraction**: Extracts only user inputs from conversations (excludes assistant responses)
3. **Deduplication**: Removes duplicate prompts
4. **Outlier Removal**: Removes text length outliers using IQR method with log transformation
5. **Caching**: Saves processed data to JSON file to prevent reprocessing

## Installation

```bash
pip install -r requirements.txt
```

Hugging Face login is required:
```bash
huggingface-cli login
```

## Usage

### Running on Host

```python
from dataset import get_prompts

# Get processed prompts
prompts = get_prompts()
print(f"Total prompts: {len(prompts)}")
```

Or run directly:
```bash
python dataset.py
```

### Using with vLLM Container

1. **Mount data directory**: Mount the host's `./data` directory to the container

```bash
docker run -v /home/xsailor6/hmchoi/ELIS/data:/app/data \
           -v /home/xsailor6/hmchoi/ELIS/dataset.py:/app/dataset.py \
           <vllm-image>
```

2. **Inside the container**:

```python
from dataset import get_prompts

# Load cached data (if processed on host)
# Or process directly inside container
prompts = get_prompts()

# Send to vLLM
for prompt in prompts:
    # Call vLLM API
    pass
```

## File Structure

```
ELIS/
├── dataset.py              # Main script
├── requirements.txt        # Dependencies
├── README.md              # Documentation
└── data/                  # Created automatically
    └── processed_dataset.json  # Processed data (auto-generated)
```

## Data Persistence

- **First run**: Download dataset → Sample → Preprocess → Save to `data/processed_dataset.json`
- **Subsequent runs**: Load cached data from `data/processed_dataset.json`

**Important**: By sharing the `data/` directory between host and container, both can use the same processed data.

## Configuration

You can modify settings at the top of `dataset.py`:

```python
SAMPLE_SIZE = 11000      # Number of samples to extract
RANDOM_SEED = 42         # Seed for reproducibility
OUTPUT_FILE = "processed_dataset.json"
DATA_DIR = "./data"      # Data storage path
```

## Processing Pipeline

1. **Sampling**: Random sampling of 11,000 examples from lmsys-chat-1m (seed=42)
2. **Extraction**: Extract only user inputs from conversations
3. **Deduplication**: Remove duplicate prompts (case-insensitive)
4. **Outlier Removal**: 
   - Apply log transformation to text lengths
   - Remove outliers using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)

## Output Format

`data/processed_dataset.json`:
```json
{
  "prompts": [
    "User prompt 1...",
    "User prompt 2...",
    ...
  ],
  "count": 10500
}
```

