"""
ELIS Data Classes

Request 및 Job 데이터 구조 정의
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid


class JobStatus(Enum):
    """Job 상태"""
    PENDING = "pending"         # 대기 중
    RUNNING = "running"         # 실행 중
    PREEMPTED = "preempted"     # 선점됨
    COMPLETED = "completed"     # 완료


@dataclass
class Request:
    """
    사용자 요청 데이터 클래스
    
    논문 Section 4.1: Frontend에서 수신하는 요청
    """
    # [NOTE, hyunnnchoi, 2025.12.01] 요청 식별자
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # [NOTE, hyunnnchoi, 2025.12.01] 입력 프롬프트
    prompt: str = ""
    
    # [NOTE, hyunnnchoi, 2025.12.01] 타임스탬프
    arrival_time: float = field(default_factory=time.time)
    
    # [NOTE, hyunnnchoi, 2025.12.01] 생성 파라미터
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __hash__(self):
        return hash(self.request_id)
    
    def __eq__(self, other):
        if isinstance(other, Request):
            return self.request_id == other.request_id
        return False


@dataclass
class Job:
    """
    스케줄링 Job 데이터 클래스
    
    논문 Section 4.1: Scheduler가 관리하는 Job 단위
    """
    # [NOTE, hyunnnchoi, 2025.12.01] Job 식별자
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # [NOTE, hyunnnchoi, 2025.12.01] 연결된 Request
    request: Optional[Request] = None
    
    # [NOTE, hyunnnchoi, 2025.12.01] Job 상태
    status: JobStatus = JobStatus.PENDING
    
    # [NOTE, hyunnnchoi, 2025.12.01] 생성된 토큰 관련
    generated_text: str = ""
    generated_tokens: int = 0
    
    # [NOTE, hyunnnchoi, 2025.12.01] 예측 관련 (논문 Section 4.2)
    predicted_remaining_tokens: float = 0.0
    last_prediction_at_tokens: int = 0  # 마지막 예측 시점의 토큰 수
    prediction_history: list = field(default_factory=list)  # 예측 히스토리
    
    # [NOTE, hyunnnchoi, 2025.12.01] 타이밍 정보
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_prefill_time: float = 0.0
    total_decode_time: float = 0.0
    
    # [NOTE, hyunnnchoi, 2025.12.01] 선점 관련 (논문 Section 4.1)
    preemption_count: int = 0
    
    @property
    def priority(self) -> float:
        """
        ISRTF 우선순위 계산
        
        낮은 값 = 높은 우선순위 (remaining tokens가 적을수록 먼저 실행)
        """
        return self.predicted_remaining_tokens
    
    @property
    def jct(self) -> Optional[float]:
        """Job Completion Time 계산"""
        if self.end_time and self.request:
            return self.end_time - self.request.arrival_time
        return None
    
    @property
    def full_context(self) -> str:
        """현재까지의 전체 컨텍스트 (프롬프트 + 생성된 텍스트)"""
        if self.request:
            if self.generated_text:
                return self.request.prompt + " " + self.generated_text
            return self.request.prompt
        return self.generated_text
    
    def needs_prediction_update(self, interval: int = 50) -> bool:
        """
        예측 업데이트가 필요한지 확인
        
        논문: 50토큰마다 재예측
        """
        tokens_since_last = self.generated_tokens - self.last_prediction_at_tokens
        return tokens_since_last >= interval
    
    def update_prediction(self, predicted_remaining: float):
        """예측값 업데이트"""
        self.predicted_remaining_tokens = predicted_remaining
        self.last_prediction_at_tokens = self.generated_tokens
        self.prediction_history.append({
            "tokens": self.generated_tokens,
            "predicted_remaining": predicted_remaining,
            "timestamp": time.time()
        })
    
    def __hash__(self):
        return hash(self.job_id)
    
    def __eq__(self, other):
        if isinstance(other, Job):
            return self.job_id == other.job_id
        return False
    
    def __lt__(self, other):
        """우선순위 비교 (heapq 사용을 위해)"""
        if isinstance(other, Job):
            return self.priority < other.priority
        return NotImplemented

