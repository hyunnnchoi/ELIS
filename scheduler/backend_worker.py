"""
ELIS Backend Worker

논문 Section 4.1: vLLM 기반 Backend Worker
- vLLM 서버와 통신
- Streaming generation 지원
- 50토큰마다 scheduler에 progress 보고
"""

import requests
import json
import time
from typing import Dict, List, Optional, Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config import ELISConfig
from .data_classes import Job, JobStatus


class BackendWorker:
    """
    vLLM Backend Worker
    
    논문 Section 4.1:
    - vLLM 서버와 HTTP 통신
    - Streaming 응답 처리
    - 토큰 단위 진행 상황 보고
    """
    
    def __init__(
        self, 
        config: ELISConfig,
        on_progress: Optional[Callable[[Job, str, int], None]] = None,
        on_complete: Optional[Callable[[Job], None]] = None
    ):
        """
        Args:
            config: ELIS 설정
            on_progress: 진행 콜백 (job, new_text, token_count)
            on_complete: 완료 콜백 (job)
        """
        self.config = config
        self.on_progress = on_progress
        self.on_complete = on_complete
        
        # [NOTE, hyunnnchoi, 2025.12.01] 토크나이저 로드 (토큰 카운트용)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.bge_model_name, 
            trust_remote_code=True
        )
        
        # [NOTE, hyunnnchoi, 2025.12.01] 실행 상태
        self.running = False
        self.executor = None
        
        print(f"[Worker] Backend Worker initialized")
        print(f"  - vLLM URL: {config.vllm_server_url}")
        print(f"  - Model: {config.model_name}")
    
    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _send_request(
        self, 
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool = False
    ) -> Optional[Dict]:
        """
        vLLM 서버에 요청 전송
        
        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰
            temperature: 샘플링 온도
            top_p: nucleus sampling
            stream: 스트리밍 모드
            
        Returns:
            API 응답 또는 None
        """
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        
        try:
            if stream:
                response = requests.post(
                    self.config.vllm_server_url,
                    json=payload,
                    stream=True,
                    timeout=300
                )
                response.raise_for_status()
                return response  # 스트림 객체 반환
            else:
                response = requests.post(
                    self.config.vllm_server_url,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.RequestException as e:
            print(f"[Worker] Request failed: {e}")
            return None
    
    def process_job(self, job: Job) -> bool:
        """
        단일 Job 처리 (Non-streaming)
        
        Args:
            job: 처리할 Job
            
        Returns:
            성공 여부
        """
        if not job.request:
            print(f"[Worker] Job {job.job_id[:8]}... has no request")
            return False
        
        request = job.request
        
        # 이미 생성된 텍스트가 있으면 프롬프트에 추가 (선점 후 재개)
        prompt = job.full_context
        remaining_tokens = request.max_tokens - job.generated_tokens
        
        if remaining_tokens <= 0:
            if self.on_complete:
                self.on_complete(job)
            return True
        
        # vLLM 요청
        response = self._send_request(
            prompt=prompt,
            max_tokens=remaining_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False
        )
        
        if not response:
            return False
        
        # 응답 처리
        if "choices" in response and len(response["choices"]) > 0:
            new_text = response["choices"][0].get("text", "")
            token_count = self._count_tokens(new_text)
            
            # 진행 콜백
            if self.on_progress:
                self.on_progress(job, new_text, token_count)
            
            # 완료 콜백
            if self.on_complete:
                self.on_complete(job)
            
            return True
        
        return False
    
    def process_job_streaming(self, job: Job) -> Generator[tuple, None, None]:
        """
        단일 Job 처리 (Streaming) - 50토큰마다 yield
        
        논문 Section 4.1: 50토큰 단위로 progress 보고
        
        Args:
            job: 처리할 Job
            
        Yields:
            (new_text, token_count) 튜플
        """
        if not job.request:
            print(f"[Worker] Job {job.job_id[:8]}... has no request")
            return
        
        request = job.request
        prompt = job.full_context
        remaining_tokens = request.max_tokens - job.generated_tokens
        
        if remaining_tokens <= 0:
            return
        
        # Streaming 요청
        response = self._send_request(
            prompt=prompt,
            max_tokens=remaining_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True
        )
        
        if not response:
            return
        
        # 스트리밍 응답 처리
        buffer = ""
        buffer_tokens = 0
        chunk_interval = self.config.prediction_interval  # 50토큰
        
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                
                line_text = line.decode('utf-8')
                if line_text.startswith("data: "):
                    data_str = line_text[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("text", "")
                            if delta:
                                buffer += delta
                                buffer_tokens = self._count_tokens(buffer)
                                
                                # 50토큰마다 yield
                                if buffer_tokens >= chunk_interval:
                                    yield (buffer, buffer_tokens)
                                    buffer = ""
                                    buffer_tokens = 0
                                    
                    except json.JSONDecodeError:
                        continue
            
            # 남은 버퍼 처리
            if buffer:
                yield (buffer, self._count_tokens(buffer))
                
        except Exception as e:
            print(f"[Worker] Streaming error: {e}")
            if buffer:
                yield (buffer, self._count_tokens(buffer))
    
    def process_batch(
        self, 
        jobs: List[Job],
        parallel: bool = True
    ) -> Dict[str, bool]:
        """
        배치 Job 처리
        
        Args:
            jobs: 처리할 Job 리스트
            parallel: 병렬 처리 여부
            
        Returns:
            {job_id: success} 딕셔너리
        """
        results = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                futures = {
                    executor.submit(self.process_job, job): job.job_id
                    for job in jobs
                }
                
                for future in as_completed(futures):
                    job_id = futures[future]
                    try:
                        success = future.result()
                        results[job_id] = success
                    except Exception as e:
                        print(f"[Worker] Error processing job {job_id[:8]}...: {e}")
                        results[job_id] = False
        else:
            for job in jobs:
                try:
                    results[job.job_id] = self.process_job(job)
                except Exception as e:
                    print(f"[Worker] Error processing job {job.job_id[:8]}...: {e}")
                    results[job.job_id] = False
        
        return results
    
    def simulate_generation(
        self, 
        job: Job, 
        target_tokens: int = 200
    ) -> Generator[tuple, None, None]:
        """
        생성 시뮬레이션 (vLLM 서버 없이 테스트용)
        
        Args:
            job: Job
            target_tokens: 생성할 총 토큰 수
            
        Yields:
            (new_text, token_count) 튜플
        """
        chunk_interval = self.config.prediction_interval
        generated = 0
        
        while generated < target_tokens:
            # 50토큰씩 생성 시뮬레이션
            chunk_size = min(chunk_interval, target_tokens - generated)
            fake_text = " token" * chunk_size  # 가짜 텍스트
            
            time.sleep(0.1)  # 생성 시간 시뮬레이션
            
            generated += chunk_size
            yield (fake_text, chunk_size)
    
    def health_check(self) -> bool:
        """
        vLLM 서버 상태 확인
        
        Returns:
            서버 정상 여부
        """
        try:
            # /health 또는 /v1/models 엔드포인트 확인
            base_url = self.config.vllm_server_url.replace("/v1/completions", "")
            response = requests.get(f"{base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            try:
                # 대안: models 엔드포인트
                base_url = self.config.vllm_server_url.replace("/completions", "/models")
                response = requests.get(base_url, timeout=5)
                return response.status_code == 200
            except:
                return False

