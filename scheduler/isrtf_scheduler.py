"""
ELIS ISRTF Scheduler

논문 Section 4.1: Iterative Shortest Remaining Time First 스케줄러
- 50토큰마다 remaining token 재예측
- 예측값 기반 우선순위 결정
- Preemption 지원
"""

import heapq
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading

from .config import ELISConfig
from .data_classes import Request, Job, JobStatus
from .predictor import ELISPredictorWrapper


class ISRTFScheduler:
    """
    ISRTF (Iterative Shortest Remaining Time First) 스케줄러
    
    논문 Section 4.1:
    - Frontend에서 요청 수신
    - 예측 모델로 remaining token 예측
    - 50토큰마다 재예측하여 우선순위 갱신
    - SRTF 기반 스케줄링
    """
    
    def __init__(self, config: ELISConfig, predictor: ELISPredictorWrapper):
        """
        Args:
            config: ELIS 설정
            predictor: 예측 모델 래퍼
        """
        self.config = config
        self.predictor = predictor
        
        # [NOTE, hyunnnchoi, 2025.12.01] Job 관리 자료구조
        self.job_queue: List[Job] = []  # Priority queue (heapq)
        self.running_jobs: Dict[str, Job] = {}  # job_id -> Job
        self.completed_jobs: Dict[str, Job] = {}  # job_id -> Job
        self.preempted_jobs: Dict[str, Job] = {}  # job_id -> Job
        
        # [NOTE, hyunnnchoi, 2025.12.01] Request-Job 매핑
        self.request_to_job: Dict[str, str] = {}  # request_id -> job_id
        self.job_to_request: Dict[str, str] = {}  # job_id -> request_id
        
        # [NOTE, hyunnnchoi, 2025.12.01] 통계
        self.stats = {
            "total_requests": 0,
            "completed_jobs": 0,
            "preemptions": 0,
            "predictions_made": 0,
        }
        
        # [NOTE, hyunnnchoi, 2025.12.01] 스레드 안전성
        self.lock = threading.RLock()
        
        print(f"[Scheduler] ISRTF Scheduler initialized")
        print(f"  - Prediction interval: {config.prediction_interval} tokens")
        print(f"  - Max batch size: {config.max_batch_size}")
    
    def submit_request(self, request: Request) -> Job:
        """
        새 Request 제출 및 Job 생성
        
        Args:
            request: 제출할 Request
            
        Returns:
            생성된 Job
        """
        with self.lock:
            # Job 생성
            job = Job(request=request)
            
            # Request-Job 매핑
            self.request_to_job[request.request_id] = job.job_id
            self.job_to_request[job.job_id] = request.request_id
            
            # 초기 예측 수행
            predicted_remaining = self.predictor.initial_predict_for_job(job)
            
            # 우선순위 큐에 추가
            heapq.heappush(self.job_queue, job)
            
            self.stats["total_requests"] += 1
            self.stats["predictions_made"] += 1
            
            print(f"[Scheduler] Request submitted: {request.request_id[:8]}... "
                  f"-> Job {job.job_id[:8]}... (predicted: {predicted_remaining:.1f} tokens)")
            
            return job
    
    def get_next_batch(self, max_size: Optional[int] = None) -> List[Job]:
        """
        다음 실행할 Job 배치 선택
        
        ISRTF: remaining token이 가장 적은 Job들 우선
        
        Args:
            max_size: 최대 배치 크기 (None이면 설정값 사용)
            
        Returns:
            실행할 Job 리스트
        """
        with self.lock:
            if max_size is None:
                max_size = self.config.max_batch_size
            
            batch = []
            
            # 우선순위 순으로 Job 선택
            while len(batch) < max_size and self.job_queue:
                job = heapq.heappop(self.job_queue)
                
                # PENDING 또는 PREEMPTED 상태만 선택
                if job.status in [JobStatus.PENDING, JobStatus.PREEMPTED]:
                    job.status = JobStatus.RUNNING
                    if job.start_time is None:
                        job.start_time = time.time()
                    self.running_jobs[job.job_id] = job
                    batch.append(job)
            
            return batch
    
    def update_job_progress(
        self, 
        job: Job, 
        new_tokens: str, 
        token_count: int
    ) -> Optional[float]:
        """
        Job 진행 상황 업데이트 및 필요시 재예측
        
        논문 Section 4.1: 50토큰마다 재예측
        
        Args:
            job: 업데이트할 Job
            new_tokens: 새로 생성된 텍스트
            token_count: 새로 생성된 토큰 수
            
        Returns:
            재예측이 수행된 경우 새 predicted_remaining, 아니면 None
        """
        with self.lock:
            # 생성 정보 업데이트
            job.generated_text += new_tokens
            job.generated_tokens += token_count
            
            # 50토큰마다 재예측
            if self.predictor.should_update_prediction(job):
                predicted_remaining = self.predictor.update_job_prediction(job)
                self.stats["predictions_made"] += 1
                
                print(f"[Scheduler] Job {job.job_id[:8]}... re-predicted: "
                      f"{predicted_remaining:.1f} remaining (at {job.generated_tokens} tokens)")
                
                return predicted_remaining
            
            return None
    
    def check_preemption(self) -> List[Tuple[Job, Job]]:
        """
        선점이 필요한 Job 쌍 확인
        
        논문 Section 4.1: SRTF 선점 로직
        
        Returns:
            (선점될 Job, 선점하는 Job) 튜플 리스트
        """
        with self.lock:
            preemptions = []
            
            # 대기 중인 Job 중 우선순위가 높은 것 확인
            if not self.job_queue or not self.running_jobs:
                return preemptions
            
            # 가장 우선순위 높은 대기 Job
            best_waiting = self.job_queue[0]  # heapq peek
            
            # 현재 실행 중인 Job 중 우선순위가 낮은 것 찾기
            for job_id, running_job in list(self.running_jobs.items()):
                if running_job.priority > best_waiting.priority:
                    # 선점 발생
                    preemptions.append((running_job, best_waiting))
            
            return preemptions
    
    def preempt_job(self, job: Job):
        """
        Job 선점 처리
        
        Args:
            job: 선점할 Job
        """
        with self.lock:
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
            
            job.status = JobStatus.PREEMPTED
            job.preemption_count += 1
            
            # 다시 큐에 추가
            heapq.heappush(self.job_queue, job)
            self.preempted_jobs[job.job_id] = job
            
            self.stats["preemptions"] += 1
            
            print(f"[Scheduler] Job {job.job_id[:8]}... preempted "
                  f"(count: {job.preemption_count})")
    
    def complete_job(self, job: Job):
        """
        Job 완료 처리
        
        Args:
            job: 완료된 Job
        """
        with self.lock:
            job.status = JobStatus.COMPLETED
            job.end_time = time.time()
            
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
            
            self.completed_jobs[job.job_id] = job
            self.stats["completed_jobs"] += 1
            
            jct = job.jct
            print(f"[Scheduler] Job {job.job_id[:8]}... completed: "
                  f"{job.generated_tokens} tokens, JCT: {jct:.2f}s")
    
    def get_job_by_request_id(self, request_id: str) -> Optional[Job]:
        """
        Request ID로 Job 조회
        
        Args:
            request_id: Request ID
            
        Returns:
            해당 Job 또는 None
        """
        with self.lock:
            job_id = self.request_to_job.get(request_id)
            if not job_id:
                return None
            
            # 각 상태별 딕셔너리에서 검색
            if job_id in self.running_jobs:
                return self.running_jobs[job_id]
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            if job_id in self.preempted_jobs:
                return self.preempted_jobs[job_id]
            
            # 큐에서 검색
            for job in self.job_queue:
                if job.job_id == job_id:
                    return job
            
            return None
    
    def get_stats(self) -> Dict:
        """
        스케줄러 통계 반환
        
        Returns:
            통계 딕셔너리
        """
        with self.lock:
            # JCT 계산
            jcts = [job.jct for job in self.completed_jobs.values() if job.jct]
            avg_jct = sum(jcts) / len(jcts) if jcts else 0.0
            
            return {
                **self.stats,
                "pending_jobs": len(self.job_queue),
                "running_jobs": len(self.running_jobs),
                "avg_jct": avg_jct,
                "total_preemptions": sum(
                    job.preemption_count for job in self.completed_jobs.values()
                ),
            }
    
    def is_empty(self) -> bool:
        """모든 Job이 완료되었는지 확인"""
        with self.lock:
            return not self.job_queue and not self.running_jobs
    
    def reset(self):
        """스케줄러 상태 초기화"""
        with self.lock:
            self.job_queue.clear()
            self.running_jobs.clear()
            self.completed_jobs.clear()
            self.preempted_jobs.clear()
            self.request_to_job.clear()
            self.job_to_request.clear()
            self.stats = {
                "total_requests": 0,
                "completed_jobs": 0,
                "preemptions": 0,
                "predictions_made": 0,
            }

