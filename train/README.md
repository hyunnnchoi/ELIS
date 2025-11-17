# ELIS Output Token Predictor - Training Module

BGE 기반 Output Token 예측 모델 트레이닝 시스템입니다.

## 📋 Overview

이 모듈은 논문 "ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor"에 설명된 Output Token Predictor를 구현합니다.

### Architecture

```
Input Text (User Prompt + Generated Text)
    ↓
BGE Model (BAAI/bge-base-en-v1.5) - Frozen
    ↓
Mean Pooling (CLS + All Tokens)
    ↓
8 Fully Connected Layers (Hidden Dim: 1024, ReLU)
    ↓
Output: Predicted Remaining Tokens (Scalar)
```

### Key Features

- **Frozen BGE Embeddings**: Pre-trained BGE model parameters are frozen
- **Mean Pooling**: Uses all tokens (including CLS) for representation
- **8 FC Layers**: Hidden dimension of 1024 with ReLU activation
- **MSE Loss**: Regression loss for token count prediction
- **Metrics**: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd train
pip install -r requirements.txt
```

### 2. Run Training

**Basic Training:**
```bash
python train.py --data-dir ../data
```

**Custom Configuration:**
```bash
python train.py \
  --data-dir ../data \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --epochs 16 \
  --hidden-dim 1024 \
  --num-layers 8 \
  --checkpoint-dir ./checkpoints
```

### 3. Monitor Training

Training will automatically:
- Split data into train/val/test (6:2:2)
- Train for specified epochs
- Save checkpoints and best model
- Evaluate on test set
- Save training history and results

## 📁 File Structure

```
train/
├── train.py          # Main training script
├── model.py          # BGE + FC layers model
├── dataset.py        # Data loading and preprocessing
├── trainer.py        # Training loop and evaluation
├── requirements.txt  # Dependencies
├── README.md         # This file
└── checkpoints/      # Auto-created for model checkpoints
    ├── best_model.pt
    ├── latest_model.pt
    ├── checkpoint_epoch_*.pt
    ├── training_history.json
    └── test_results.json
```

## ⚙️ Configuration

### Command Line Arguments

#### Data Arguments
- `--data-dir`: Path to data directory (default: `../data`)
- `--max-length`: Maximum sequence length for BGE (default: `512`)

#### Model Arguments
- `--bge-model`: BGE model name (default: `BAAI/bge-base-en-v1.5`)
- `--hidden-dim`: Hidden dimension for FC layers (default: `1024`)
- `--num-layers`: Number of FC layers (default: `8`)
- `--no-freeze-bge`: Fine-tune BGE parameters instead of freezing

#### Training Arguments
- `--batch-size`: Training batch size (default: `16`)
- `--learning-rate`: Learning rate (default: `1e-4`)
- `--epochs`: Number of training epochs (default: `16`)
- `--early-stopping`: Early stopping patience (default: `5`)
- `--num-workers`: Number of data loader workers (default: `4`)

#### Checkpoint Arguments
- `--checkpoint-dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--resume`: Path to checkpoint to resume from
- `--save-every`: Save checkpoint every N epochs (default: `1`)

#### Other Arguments
- `--seed`: Random seed for reproducibility (default: `42`)
- `--device`: Device to train on (default: auto-detect cuda/cpu)
- `--log-interval`: Logging interval in batches (default: `100`)

## 📊 Training Details

### Dataset

- **Source**: `ELIS/data/{model}/vllm_results_training.jsonl`
- **Models**: llama2-7b-hf, llama2-13b-hf, gpt-oss-20b, opt-6.7b, opt-13b, vicuna-13b-v1.5
- **Split**: 60% train, 20% validation, 20% test
- **Total Samples**: ~100K+ (varies by model)

### Data Format

Each training sample:
```json
{
  "input_prompt": "User question...",
  "output_prompt": "Generated text so far...",
  "number_of_output_tokens": 100,
  "remaining_tokens": 50  // LABEL
}
```

**Input**: `input_prompt + output_prompt` (full context seen by model)  
**Label**: `remaining_tokens` (how many tokens left to generate)

### Hyperparameters (논문 기준)

| Parameter | Value |
|-----------|-------|
| Base Model | BAAI/bge-base-en-v1.5 |
| BGE Parameters | Frozen |
| FC Hidden Dim | 1024 |
| Number of FC Layers | 8 |
| Activation | ReLU |
| Loss Function | MSE |
| Optimizer | Adam |
| Learning Rate | 1×10⁻⁴ |
| Batch Size | 16 |
| Epochs | 16 |
| Dataset Split | 6:2:2 |

## 📈 Output Files

### Checkpoints

- `best_model.pt`: Best model based on validation loss
- `latest_model.pt`: Most recent model checkpoint
- `checkpoint_epoch_N.pt`: Checkpoint at epoch N

### Training History

`training_history.json`:
```json
{
  "train_loss": [...],
  "train_mae": [...],
  "train_rmse": [...],
  "val_loss": [...],
  "val_mae": [...],
  "val_rmse": [...],
  "epoch_times": [...]
}
```

### Test Results

`test_results.json`:
```json
{
  "test_loss": 0.1234,
  "test_mae": 5.67,
  "test_rmse": 8.90,
  "best_epoch": 12,
  "best_val_loss": 0.1150
}
```

## 🔧 Advanced Usage

### Resume Training

```bash
python train.py --resume ./checkpoints/latest_model.pt
```

### Fine-tune BGE Model

```bash
python train.py --no-freeze-bge
```

### Custom Model Configuration

```bash
python train.py \
  --hidden-dim 2048 \
  --num-layers 12 \
  --learning-rate 5e-5
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch-size 64
```

## 📊 Expected Results

논문에 보고된 성능:
- Training converges around **epoch 16**
- MAE: ~**5-10 tokens** (모델에 따라 다름)
- RMSE: ~**10-20 tokens** (모델에 따라 다름)

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --batch-size 8

# Reduce max sequence length
python train.py --max-length 256
```

### Slow Training
```bash
# Increase number of workers
python train.py --num-workers 8

# Enable pin_memory (automatic for CUDA)
```

### Model Not Converging
```bash
# Adjust learning rate
python train.py --learning-rate 5e-5

# Increase epochs
python train.py --epochs 32
```

## 📝 Notes

- BGE 모델은 기본적으로 frozen되어 있으며, 오직 8개의 FC layer만 학습됩니다.
- Mean pooling은 CLS token과 모든 다른 토큰들을 포함합니다.
- 데이터셋은 여러 LLM 모델들의 출력을 통합하여 사용합니다.
- Random seed (42)를 사용하여 재현 가능한 결과를 보장합니다.

## 📚 Citation

```bibtex
@misc{choi2025elisefficientllmiterative,
      title={ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor}, 
      author={Seungbeom Choi and Jeonghoe Goo and Eunjoo Jeon and Mingyu Yang and Minsung Jang},
      year={2025},
      eprint={2505.09142},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2505.09142}, 
}
```

