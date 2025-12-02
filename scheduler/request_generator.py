"""
ELIS Request Generator

논문 Section 6.1: Gamma distribution 기반 request arrival 시뮬레이션
LMSYS-Chat-1M에서 프롬프트 샘플링
"""

import json
import random
import numpy as np
from typing import List, Generator
import time

from .config import ELISConfig
from .data_classes import Request


class RequestGenerator:
    """
    Request 생성기
    
    논문 Section 6.1 Methodology:
    - Gamma distribution으로 request arrival 시뮬레이션
    - LMSYS-Chat-1M에서 프롬프트 샘플링
    """
    
    def __init__(self, config: ELISConfig):
        """
        Args:
            config: ELIS 설정
        """
        self.config = config
        
        # [NOTE, hyunnnchoi, 2025.12.01] Random seed 설정
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # [NOTE, hyunnnchoi, 2025.12.01] 프롬프트 로드 및 샘플링
        self.prompts = self._load_and_sample_prompts()
        self.prompt_index = 0
        
        # [NOTE, hyunnnchoi, 2025.12.01] Gamma distribution 파라미터 (Section 6.1)
        self.gamma_shape = config.gamma_shape
        self.gamma_scale = config.gamma_scale
        
        print(f"[RequestGenerator] Initialized with:")
        print(f"  - Prompts loaded: {len(self.prompts)}")
        print(f"  - Gamma distribution: shape={self.gamma_shape}, scale={self.gamma_scale}")
    
    def _load_and_sample_prompts(self) -> List[str]:
        """
        프롬프트 로드 및 샘플링
        
        논문: LMSYS-Chat-1M에서 200개 별도 샘플링
        """
        try:
            with open(self.config.prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_prompts = data.get('prompts', [])
            
            # 200개 샘플링 (또는 설정된 수)
            num_samples = min(self.config.num_eval_prompts, len(all_prompts))
            sampled_prompts = random.sample(all_prompts, num_samples)
            
            print(f"[RequestGenerator] Sampled {num_samples} prompts from {len(all_prompts)} total")
            
            return sampled_prompts
            
        except FileNotFoundError:
            print(f"[RequestGenerator] Warning: Prompts file not found: {self.config.prompts_file}")
            print("[RequestGenerator] Using default test prompts")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> List[str]:
        """기본 테스트 프롬프트"""
        return [
            "What is the capital of France?",
            "Explain quantum mechanics in simple terms.",
            "Write a short story about a robot.",
            "How do I make a chocolate cake?",
            "What are the benefits of exercise?",
            "Explain the theory of relativity.",
            "Write a poem about the ocean.",
            "What is machine learning?",
            "How does a computer work?",
            "Tell me about climate change.",
        ] * 20  # 200개로 확장
    
    def get_inter_arrival_time(self) -> float:
        """
        Gamma distribution에서 inter-arrival time 샘플링
        
        논문 Section 6.1: Request arrival은 Gamma distribution을 따름
        
        Returns:
            다음 요청까지의 시간 (초)
        """
        return np.random.gamma(self.gamma_shape, self.gamma_scale)
    
    def get_next_prompt(self) -> str:
        """
        다음 프롬프트 반환 (순환)
        
        Returns:
            프롬프트 문자열
        """
        prompt = self.prompts[self.prompt_index % len(self.prompts)]
        self.prompt_index += 1
        return prompt
    
    def create_request(self) -> Request:
        """
        새 Request 생성
        
        Returns:
            생성된 Request 객체
        """
        return Request(
            prompt=self.get_next_prompt(),
            arrival_time=time.time(),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
    
    def generate_requests(self, num_requests: int) -> Generator[Request, None, None]:
        """
        지정된 수의 Request를 Gamma distribution 간격으로 생성
        
        Args:
            num_requests: 생성할 요청 수
            
        Yields:
            생성된 Request 객체
        """
        for i in range(num_requests):
            # Request 생성
            request = self.create_request()
            yield request
            
            # 다음 요청까지 대기 (마지막 요청 제외)
            if i < num_requests - 1:
                inter_arrival_time = self.get_inter_arrival_time()
                time.sleep(inter_arrival_time)
    
    def generate_requests_async(self, num_requests: int) -> List[tuple]:
        """
        비동기 시뮬레이션을 위한 Request 및 arrival time 생성
        
        실제 대기 없이 arrival time만 계산
        
        Args:
            num_requests: 생성할 요청 수
            
        Returns:
            (Request, arrival_time) 튜플 리스트
        """
        requests = []
        current_time = 0.0
        
        for _ in range(num_requests):
            # Request 생성 (arrival_time은 시뮬레이션 시간으로 설정)
            request = Request(
                prompt=self.get_next_prompt(),
                arrival_time=current_time,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            requests.append((request, current_time))
            
            # 다음 arrival time 계산
            inter_arrival_time = self.get_inter_arrival_time()
            current_time += inter_arrival_time
        
        return requests
    
    def reset(self):
        """생성기 상태 초기화"""
        self.prompt_index = 0
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

