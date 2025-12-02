"""
ELIS Predictor Wrapper

논문 Section 4.2: BGE 기반 Response Length Prediction Model
50토큰 단위로 iterative prediction 수행
"""

import torch
from transformers import AutoTokenizer
from typing import Optional
import sys
import os

# train 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'train'))
from model import ELISPredictor

from .config import ELISConfig
from .data_classes import Job


class ELISPredictorWrapper:
    """
    ELIS Response Length Predictor 래퍼 클래스
    
    논문 Section 4.2:
    - BGE (BAAI/bge-base-en-v1.5) 기반 임베딩
    - 8개의 FC 레이어
    - Mean pooling 사용
    - 50토큰마다 재예측
    """
    
    def __init__(self, config: ELISConfig):
        """
        Args:
            config: ELIS 설정
        """
        self.config = config
        self.device = config.device
        self.prediction_interval = config.prediction_interval
        
        # [NOTE, hyunnnchoi, 2025.12.01] 모델 초기화
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(config.bge_model_name)
        
        print(f"[Predictor] Initialized with:")
        print(f"  - Device: {self.device}")
        print(f"  - Prediction interval: {self.prediction_interval} tokens")
    
    def _load_model(self) -> ELISPredictor:
        """체크포인트에서 모델 로드"""
        print(f"[Predictor] Loading model from {self.config.predictor_checkpoint}...")
        
        # 모델 초기화
        model = ELISPredictor(
            bge_model_name=self.config.bge_model_name,
            hidden_dim=self.config.predictor_hidden_dim,
            num_layers=self.config.predictor_num_layers,
            freeze_bge=True
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(
            self.config.predictor_checkpoint, 
            map_location=self.device,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"[Predictor] Model loaded (Epoch: {epoch})")
        
        return model
    
    def predict(self, text: str) -> float:
        """
        단일 텍스트에 대한 remaining token 예측
        
        Args:
            text: 프롬프트 + 현재까지 생성된 텍스트
            
        Returns:
            예측된 remaining token 수
        """
        # 토크나이즈
        encoded = self.tokenizer(
            text,
            max_length=self.config.predictor_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            prediction = self.model(input_ids, attention_mask)
        
        # 음수 예측은 0으로 클램프
        return max(0.0, prediction.item())
    
    def predict_batch(self, texts: list) -> list:
        """
        배치 텍스트에 대한 remaining token 예측
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            예측된 remaining token 수 리스트
        """
        if not texts:
            return []
        
        # 배치 토크나이즈
        encoded = self.tokenizer(
            texts,
            max_length=self.config.predictor_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            predictions = self.model(input_ids, attention_mask)
        
        # 음수 예측은 0으로 클램프
        return [max(0.0, p.item()) for p in predictions]
    
    def update_job_prediction(self, job: Job) -> float:
        """
        Job의 remaining token 예측 업데이트
        
        논문 Section 4.2: 50토큰마다 재예측 수행
        
        Args:
            job: 업데이트할 Job
            
        Returns:
            예측된 remaining token 수
        """
        # 전체 컨텍스트로 예측
        full_context = job.full_context
        predicted_remaining = self.predict(full_context)
        
        # Job 예측값 업데이트
        job.update_prediction(predicted_remaining)
        
        return predicted_remaining
    
    def should_update_prediction(self, job: Job) -> bool:
        """
        Job의 예측 업데이트가 필요한지 확인
        
        Args:
            job: 확인할 Job
            
        Returns:
            업데이트 필요 여부
        """
        return job.needs_prediction_update(self.prediction_interval)
    
    def initial_predict_for_job(self, job: Job) -> float:
        """
        새 Job에 대한 초기 예측 수행
        
        Args:
            job: 새 Job
            
        Returns:
            예측된 remaining token 수
        """
        # 프롬프트만으로 초기 예측
        if job.request:
            predicted_remaining = self.predict(job.request.prompt)
        else:
            predicted_remaining = self.predict(job.generated_text or "")
        
        # Job 예측값 초기화
        job.predicted_remaining_tokens = predicted_remaining
        job.last_prediction_at_tokens = 0
        job.prediction_history.append({
            "tokens": 0,
            "predicted_remaining": predicted_remaining,
            "timestamp": None  # 초기 예측
        })
        
        return predicted_remaining

