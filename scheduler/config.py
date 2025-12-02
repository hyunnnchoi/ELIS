"""
ELIS Configuration Module

논문 Section 4.1, 6.1 기반 설정값 정의
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ELISConfig:
    """
    ELIS 시스템 설정
    
    논문 참조: https://arxiv.org/html/2505.09142v1
    """
    
    # [NOTE, hyunnnchoi, 2025.12.01] vLLM 서버 설정
    vllm_server_url: str = "http://localhost:8000/v1/completions"
    model_name: str = "meta-llama/Llama-2-7b-hf"
    
    # [NOTE, hyunnnchoi, 2025.12.01] 예측 모델 설정 (Section 4.2)
    predictor_checkpoint: str = "./train/checkpoints/latest_model.pt"
    bge_model_name: str = "BAAI/bge-base-en-v1.5"
    predictor_hidden_dim: int = 1024
    predictor_num_layers: int = 8
    predictor_max_length: int = 512
    
    # [NOTE, hyunnnchoi, 2025.12.01] 스케줄링 설정 (Section 4.1)
    prediction_interval: int = 50  # 50토큰마다 재예측
    max_batch_size: int = 32
    
    # [NOTE, hyunnnchoi, 2025.12.01] Request Generator 설정 (Section 6.1)
    # Gamma distribution parameters for request arrival
    gamma_shape: float = 1.0  # shape parameter (k)
    gamma_scale: float = 1.0  # scale parameter (θ), mean = k * θ
    
    # [NOTE, hyunnnchoi, 2025.12.01] 프롬프트 설정
    num_eval_prompts: int = 200  # LMSYS에서 샘플링할 프롬프트 수
    prompts_file: str = "./data/processed_dataset.json"
    
    # [NOTE, hyunnnchoi, 2025.12.01] 생성 설정
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    
    # [NOTE, hyunnnchoi, 2025.12.01] 시스템 설정
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 10
    
    # [NOTE, hyunnnchoi, 2025.12.01] 실험 설정
    total_requests: int = 1000  # 총 요청 수
    warmup_requests: int = 10   # 워밍업 요청 수
    
    def __post_init__(self):
        """설정값 검증"""
        assert self.prediction_interval > 0, "prediction_interval must be positive"
        assert self.max_batch_size > 0, "max_batch_size must be positive"
        assert self.gamma_shape > 0, "gamma_shape must be positive"
        assert self.gamma_scale > 0, "gamma_scale must be positive"

