# ELIS: Embedding Learning Input-output Sequences

A complete pipeline for generating training data for embedding models from LLM inference outputs. This project processes the lmsys-chat-1m dataset, generates completions using vLLM, and creates training data by slicing outputs into progressive token chunks.

## Overview

ELIS provides a two-stage pipeline:
1. **Dataset Processing** (`lmsys-dataset/`): Sample and preprocess user prompts from lmsys-chat-1m
2. **vLLM Request Generation** (`vllm-request-generator/`): Send prompts to vLLM and generate training data

The training data format creates progressive output sequences (e.g., 0-50 tokens, 0-100 tokens, etc.) from each LLM completion, useful for training embedding models to predict continuation lengths.

## Features

### Dataset Processing
- **Deterministic Sampling**: Fixed random seed (42) for reproducible 11,000 sample selection
- **Language Filtering**: Optional English-only filtering for BGE model compatibility
- **User Input Extraction**: Extracts only first user inputs from conversations
- **Deduplication**: Removes duplicate prompts (case-insensitive)
- **Outlier Removal**: IQR method with log transformation for text length filtering
- **Caching**: Saves processed data to prevent reprocessing

### vLLM Request Generation
- **Parallel Processing**: ThreadPoolExecutor-based batch processing (default: 16 concurrent requests)
- **Progressive Token Slicing**: Automatically slices outputs into 50-token cumulative chunks
- **Training Data Generation**: JSONL format with input/output pairs and token counts
- **Model Flexibility**: Supports multiple LLM models
- **Real-time Progress**: Live progress bars and logging
- **Robust Error Handling**: Continues processing on individual failures

## Installation

```bash
pip install -r requirements.txt
```

Hugging Face login is required for dataset access:
```bash
huggingface-cli login
```

## Quick Start

### 1. Process Dataset

```bash
cd lmsys-dataset
python dataset.py
```

This will:
- Download lmsys-chat-1m dataset
- Sample 11,000 English prompts
- Apply preprocessing (deduplication, outlier removal)
- Save to `../data/processed_dataset.json`

### 2. Generate vLLM Completions

```bash
cd vllm-request-generator
python send_to_vllm.py \
  --input /data/processed_dataset.json \
  --output /data/model-name/vllm_results.json \
  --server-url http://localhost:8000/v1/completions \
  --model model-name \
  --batch-size 16
```

This will:
- Send prompts to vLLM server in parallel batches
- Slice each completion into 50-token progressive chunks
- Save training data to `vllm_results_training.jsonl`
- Save full results to `vllm_results.json`

## Project Structure

```
ELIS/
├── README.md                          # This file
├── requirements.txt                   # Project dependencies
│
├── lmsys-dataset/                     # Dataset processing module
│   ├── dataset.py                     # Main preprocessing script
│   ├── inference_example.py           # Usage examples
│   ├── docker-compose.yml             # Container setup
│   ├── requirements.txt               # Module dependencies
│   └── README.md                      # Module documentation
│
├── vllm-request-generator/            # vLLM interaction module
│   ├── send_to_vllm.py               # Request generation script
│   ├── requirements.txt               # Module dependencies
│   └── README.md                      # Module documentation
│
├── data/                              # Data storage (auto-generated)
│   ├── processed_dataset.json         # Preprocessed prompts (EN)
│   ├── processed_dataset_kr.json      # Preprocessed prompts (KR)
│   │
│   └── {model-name}/                  # Per-model results
│       ├── vllm_results.json          # Full API responses
│       ├── vllm_results_training.jsonl # Training data (JSONL)
│       └── vllm_run.log               # Execution logs
│
└── venv/                              # Python virtual environment
```

### Supported Models

The `data/` directory contains results from multiple models:
- `gpt-oss-20b/`
- `llama2-7b-hf/`, `llama2-13b-hf/`
- `opt-6.7b/`, `opt-13b/`
- `vicuna-13b-v1.5/`

## Usage Details

### Dataset Processing Configuration

Edit `lmsys-dataset/dataset.py`:

```python
DATASET_NAME = "lmsys/lmsys-chat-1m"
SAMPLE_SIZE = 11000
RANDOM_SEED = 42
FILTER_ENGLISH_ONLY = True  # Filter for English prompts
OUTPUT_FILE = "processed_dataset.json"
DATA_DIR = "./data"
```

### vLLM Request Parameters

```bash
python send_to_vllm.py \
  --input /data/processed_dataset.json \
  --output /data/model-name/vllm_results.json \
  --server-url http://localhost:8000/v1/completions \
  --model model-name \
  --max-tokens 1024 \
  --temperature 0.7 \
  --batch-size 16 \
  --tokenizer-path /model
```

**Parameters:**
- `--input`: Input JSON file path (default: `/data/processed_dataset.json`)
- `--output`: Output JSON file path (default: `/data/vllm_results.json`)
- `--server-url`: vLLM server URL (default: `http://localhost:8000/v1/completions`)
- `--model`: Model name to use
- `--max-tokens`: Maximum tokens to generate (default: None, unlimited)
- `--temperature`: Sampling temperature (default: 0.7)
- `--batch-size`: Concurrent requests (default: 16)
- `--tokenizer-path`: Tokenizer path (default: `/model`)
- `--delay`: Delay between requests in seconds (default: 0.0)

### Background Execution

For long-running jobs:

```bash
nohup python3 send_to_vllm.py \
  --input /data/processed_dataset.json \
  --output /data/model-name/vllm_results.json \
  --model model-name \
  --batch-size 16 \
  > /data/model-name/vllm_run.log 2>&1 &

echo "PID: $!"
tail -f /data/model-name/vllm_run.log
```

## Output Formats

### Training Data (JSONL)

`data/{model-name}/vllm_results_training.jsonl`:

```jsonl
{"input_prompt": "User prompt...", "output_prompt": "0-50 token response", "number_of_output_tokens": 50, "remaining_tokens": 200}
{"input_prompt": "User prompt...", "output_prompt": "0-100 token response", "number_of_output_tokens": 100, "remaining_tokens": 150}
{"input_prompt": "User prompt...", "output_prompt": "0-150 token response", "number_of_output_tokens": 150, "remaining_tokens": 100}
```

Each prompt generates multiple training samples with progressively longer outputs (50-token increments).

### Full Results (JSON)

`data/{model-name}/vllm_results.json`:

```json
{
  "total": 10000,
  "success": 9998,
  "failed": 2,
  "total_training_samples": 49990,
  "results": [
    {
      "index": 0,
      "input_prompt": "User prompt...",
      "output_text": "Full LLM response...",
      "full_response": { /* vLLM API response */ },
      "status": "success"
    }
  ]
}
```

## Processing Pipeline

### Stage 1: Dataset Processing

1. **Load Dataset**: Download lmsys-chat-1m from Hugging Face
2. **Filter Language**: Keep only English conversations (optional)
3. **Sample**: Randomly select 11,000 examples (seed=42)
4. **Extract**: Extract first user input from each conversation
5. **Deduplicate**: Remove duplicate prompts (case-insensitive)
6. **Remove Outliers**: IQR method with log transformation on text lengths
7. **Cache**: Save to `processed_dataset.json`

### Stage 2: vLLM Generation

1. **Load Prompts**: Read from `processed_dataset.json`
2. **Parallel Requests**: Send prompts in batches to vLLM
3. **Collect Responses**: Gather completions from vLLM API
4. **Token Slicing**: Split each output into 50-token progressive chunks
5. **Save Training Data**: Write JSONL with input-output-token triplets
6. **Save Full Results**: Write complete responses for debugging

## Docker Integration

Using with vLLM container:

```bash
docker run -v /path/to/ELIS/data:/data \
           -v /path/to/ELIS/vllm-request-generator:/app \
           <vllm-image> \
           python /app/send_to_vllm.py --model model-name
```

## Monitoring

Watch real-time progress:

```bash
# Monitor logs
tail -f data/model-name/vllm_run.log

# Count training samples
watch -n 5 'wc -l data/model-name/vllm_results_training.jsonl'
```

## Notes

- **Tokenization**: Uses model-specific tokenizer for accurate token counting
- **Error Handling**: Failed requests are logged but don't stop processing
- **Memory Efficiency**: Streaming JSONL output prevents memory overflow
- **Reproducibility**: Fixed random seed ensures consistent dataset sampling

