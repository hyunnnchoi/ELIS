# ELIS Scheduler

ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor

논문: https://arxiv.org/abs/2505.09142

## 📋 Overview

이 모듈은 ELIS 논문의 ISRTF (Iterative Shortest Remaining Time First) 스케줄러를 구현합니다.

### 핵심 기능

1. **Response Length Prediction**: BGE 기반 예측 모델로 remaining token 예측
2. **ISRTF Scheduling**: 50토큰마다 재예측하여 우선순위 갱신
3. **Gamma Distribution Request**: 현실적인 request arrival 시뮬레이션
4. **vLLM Integration**: vLLM 서버와 연동

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ELIS Scheduler                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Request   │───▶│    ISRTF    │───▶│   Backend   │     │
│  │  Generator  │    │  Scheduler  │    │   Worker    │     │
│  │  (Gamma)    │    │             │    │   (vLLM)    │     │
│  └─────────────┘    └──────┬──────┘    └─────────────┘     │
│                            │                                 │
│                     ┌──────▼──────┐                         │
│                     │  Predictor  │                         │
│                     │    (BGE)    │                         │
│                     └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Simulation Mode (vLLM 서버 없이)

```bash
cd /home/work/hyunmokchoi/ELIS
python -m scheduler.run_elis --mode simulation --num-requests 50
```

### 2. Live Mode (vLLM 서버 연동)

```bash
# vLLM 서버가 실행 중이어야 함
python -m scheduler.run_elis \
  --mode live \
  --vllm-url http://localhost:8000/v1/completions \
  --model meta-llama/Llama-2-7b-hf \
  --num-requests 100
```

## 📁 File Structure

```
scheduler/
├── __init__.py           # Module exports
├── config.py             # Configuration dataclass
├── data_classes.py       # Request, Job, JobStatus
├── predictor.py          # BGE-based predictor wrapper
├── request_generator.py  # Gamma distribution request generator
├── isrtf_scheduler.py    # ISRTF scheduler implementation
├── backend_worker.py     # vLLM backend worker
├── run_elis.py          # Main runner script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## ⚙️ Configuration

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | simulation | `simulation` or `live` |
| `--vllm-url` | http://localhost:8000/v1/completions | vLLM server URL |
| `--model` | meta-llama/Llama-2-7b-hf | Model name |
| `--num-requests` | 100 | Number of requests |
| `--batch-size` | 32 | Maximum batch size |
| `--prediction-interval` | 50 | Re-prediction interval (tokens) |
| `--gamma-shape` | 1.0 | Gamma distribution shape |
| `--gamma-scale` | 1.0 | Gamma distribution scale |
| `--checkpoint` | ./train/checkpoints/latest_model.pt | Predictor checkpoint |
| `--prompts-file` | ./data/processed_dataset.json | Prompts file |
| `--output` | ./scheduler/results/elis_results.json | Output file |

### Gamma Distribution

논문 Section 6.1에 따라 request arrival은 Gamma distribution을 따릅니다:

```
inter_arrival_time ~ Gamma(shape, scale)
mean = shape * scale
```

## 📊 Output

결과는 JSON 형식으로 저장됩니다:

```json
{
  "config": { ... },
  "start_time": "2025-12-01T...",
  "end_time": "2025-12-01T...",
  "elapsed_time": 123.45,
  "final_stats": {
    "total_requests": 100,
    "completed_jobs": 100,
    "avg_jct": 2.34,
    "preemptions": 15,
    "predictions_made": 450
  },
  "jobs": [
    {
      "job_id": "...",
      "request_id": "...",
      "generated_tokens": 250,
      "jct": 2.1,
      "preemption_count": 0,
      "predicted_remaining_history": [...]
    }
  ]
}
```

## 🔧 Key Components

### 1. ELISPredictorWrapper

50토큰 단위로 remaining token 예측을 수행합니다.

```python
from scheduler import ELISPredictorWrapper, ELISConfig

config = ELISConfig(predictor_checkpoint="./train/checkpoints/latest_model.pt")
predictor = ELISPredictorWrapper(config)

# 단일 예측
remaining = predictor.predict("What is the capital of France?")

# Job 예측 업데이트
predictor.update_job_prediction(job)
```

### 2. ISRTFScheduler

ISRTF 스케줄링을 수행합니다.

```python
from scheduler import ISRTFScheduler

scheduler = ISRTFScheduler(config, predictor)

# Request 제출
job = scheduler.submit_request(request)

# 다음 배치 선택
batch = scheduler.get_next_batch(max_size=8)

# Progress 업데이트 (50토큰마다 재예측)
scheduler.update_job_progress(job, new_text, token_count)

# 선점 확인
preemptions = scheduler.check_preemption()
```

### 3. RequestGenerator

Gamma distribution 기반 request 생성을 수행합니다.

```python
from scheduler import RequestGenerator

generator = RequestGenerator(config)

# 단일 request
request = generator.create_request()

# 비동기 시뮬레이션용
requests_with_times = generator.generate_requests_async(100)
```

## 📝 Notes

- 예측 모델은 `train/` 디렉토리에서 학습된 체크포인트를 사용합니다.
- Simulation 모드는 vLLM 서버 없이 테스트할 수 있습니다.
- Live 모드는 실제 vLLM 서버가 필요합니다.

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

