#!/usr/bin/env python3
"""
ELIS Scheduler Main Runner

ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor
논문: https://arxiv.org/abs/2505.09142

실행 모드:
1. simulation: vLLM 서버 없이 시뮬레이션
2. live: 실제 vLLM 서버와 연동
"""

import argparse
import time
import json
import sys
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from scheduler.config import ELISConfig
from scheduler.data_classes import Request, Job, JobStatus
from scheduler.predictor import ELISPredictorWrapper
from scheduler.request_generator import RequestGenerator
from scheduler.isrtf_scheduler import ISRTFScheduler
from scheduler.backend_worker import BackendWorker


class ELISRunner:
    """
    ELIS 시스템 통합 러너
    
    논문 Section 4.1의 전체 플로우 구현:
    1. Request 수신 (Gamma distribution)
    2. Job 생성 및 초기 예측
    3. ISRTF 스케줄링
    4. Backend 실행
    5. 50토큰마다 재예측 및 우선순위 갱신
    """
    
    def __init__(self, config: ELISConfig):
        """
        Args:
            config: ELIS 설정
        """
        self.config = config
        
        # [NOTE, hyunnnchoi, 2025.12.01] 컴포넌트 초기화
        print("=" * 60)
        print("ELIS: Efficient LLM Iterative Scheduling System")
        print("=" * 60)
        print()
        
        print("[1/4] Initializing Predictor...")
        self.predictor = ELISPredictorWrapper(config)
        
        print("\n[2/4] Initializing Request Generator...")
        self.request_generator = RequestGenerator(config)
        
        print("\n[3/4] Initializing ISRTF Scheduler...")
        self.scheduler = ISRTFScheduler(config, self.predictor)
        
        print("\n[4/4] Initializing Backend Worker...")
        self.worker = BackendWorker(
            config,
            on_progress=self._on_job_progress,
            on_complete=self._on_job_complete
        )
        
        # [NOTE, hyunnnchoi, 2025.12.01] 결과 저장
        self.results = {
            "config": self._config_to_dict(config),
            "start_time": None,
            "end_time": None,
            "jobs": [],
        }
        
        print("\n" + "=" * 60)
        print("ELIS Initialized Successfully!")
        print("=" * 60 + "\n")
    
    def _config_to_dict(self, config: ELISConfig) -> Dict:
        """Config를 딕셔너리로 변환"""
        return {
            "vllm_server_url": config.vllm_server_url,
            "model_name": config.model_name,
            "prediction_interval": config.prediction_interval,
            "max_batch_size": config.max_batch_size,
            "gamma_shape": config.gamma_shape,
            "gamma_scale": config.gamma_scale,
            "num_eval_prompts": config.num_eval_prompts,
            "total_requests": config.total_requests,
        }
    
    def _on_job_progress(self, job: Job, new_text: str, token_count: int):
        """Job 진행 콜백"""
        # 스케줄러에 progress 보고
        predicted = self.scheduler.update_job_progress(job, new_text, token_count)
        
        if predicted is not None:
            # 재예측이 수행됨 - 선점 확인
            preemptions = self.scheduler.check_preemption()
            for preempted_job, preempting_job in preemptions:
                print(f"[Runner] Preemption detected: "
                      f"{preempted_job.job_id[:8]}... <- {preempting_job.job_id[:8]}...")
    
    def _on_job_complete(self, job: Job):
        """Job 완료 콜백"""
        self.scheduler.complete_job(job)
        
        # 결과 저장
        self.results["jobs"].append({
            "job_id": job.job_id,
            "request_id": job.request.request_id if job.request else None,
            "prompt": job.request.prompt[:100] if job.request else "",
            "generated_tokens": job.generated_tokens,
            "predicted_remaining_history": job.prediction_history,
            "jct": job.jct,
            "preemption_count": job.preemption_count,
        })
    
    def run_simulation(self, num_requests: int = 100):
        """
        시뮬레이션 모드 실행 (vLLM 서버 없이)
        
        Args:
            num_requests: 처리할 요청 수
        """
        print(f"\n[Simulation Mode] Running with {num_requests} requests...")
        print("-" * 60)
        
        self.results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        # Request 생성 (비동기 시뮬레이션용)
        requests_with_times = self.request_generator.generate_requests_async(num_requests)
        
        current_sim_time = 0.0
        request_idx = 0
        
        while request_idx < len(requests_with_times) or not self.scheduler.is_empty():
            # 새 Request 도착 처리
            while (request_idx < len(requests_with_times) and 
                   requests_with_times[request_idx][1] <= current_sim_time):
                request, arrival_time = requests_with_times[request_idx]
                self.scheduler.submit_request(request)
                request_idx += 1
            
            # 배치 선택 및 실행
            batch = self.scheduler.get_next_batch(max_size=1)  # 시뮬레이션에서는 1개씩
            
            for job in batch:
                # 생성 시뮬레이션 (50토큰씩)
                target_tokens = 100 + (hash(job.job_id) % 300)  # 100-400 토큰
                
                for new_text, token_count in self.worker.simulate_generation(job, target_tokens):
                    self._on_job_progress(job, new_text, token_count)
                    
                    # 선점 체크
                    if job.status == JobStatus.PREEMPTED:
                        break
                
                # 완료 처리 (선점되지 않은 경우)
                if job.status == JobStatus.RUNNING:
                    self._on_job_complete(job)
            
            # 시뮬레이션 시간 진행
            current_sim_time += 0.1
            
            # 진행 상황 출력
            if self.scheduler.stats["completed_jobs"] % 10 == 0:
                stats = self.scheduler.get_stats()
                print(f"\r[Progress] Completed: {stats['completed_jobs']}/{num_requests}, "
                      f"Pending: {stats['pending_jobs']}, "
                      f"Preemptions: {stats['preemptions']}", end="")
        
        end_time = time.time()
        self.results["end_time"] = datetime.now().isoformat()
        
        print("\n" + "-" * 60)
        self._print_final_stats(end_time - start_time)
    
    def run_live(self, num_requests: int = 100):
        """
        라이브 모드 실행 (실제 vLLM 서버 연동)
        
        Args:
            num_requests: 처리할 요청 수
        """
        print(f"\n[Live Mode] Running with {num_requests} requests...")
        
        # vLLM 서버 상태 확인
        if not self.worker.health_check():
            print("[Error] vLLM server is not responding!")
            print(f"  URL: {self.config.vllm_server_url}")
            return
        
        print("[OK] vLLM server is healthy")
        print("-" * 60)
        
        self.results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        # Request 생성 및 처리
        for i, request in enumerate(self.request_generator.generate_requests(num_requests)):
            # 스케줄러에 제출
            job = self.scheduler.submit_request(request)
            
            # 즉시 실행 (단순화된 버전)
            batch = self.scheduler.get_next_batch(max_size=1)
            
            for job in batch:
                # Streaming 실행
                for new_text, token_count in self.worker.process_job_streaming(job):
                    self._on_job_progress(job, new_text, token_count)
                
                # 완료
                if job.status == JobStatus.RUNNING:
                    self._on_job_complete(job)
            
            # 진행 상황
            if (i + 1) % self.config.log_interval == 0:
                stats = self.scheduler.get_stats()
                print(f"[Progress] {i+1}/{num_requests} requests, "
                      f"Avg JCT: {stats['avg_jct']:.2f}s")
        
        end_time = time.time()
        self.results["end_time"] = datetime.now().isoformat()
        
        print("-" * 60)
        self._print_final_stats(end_time - start_time)
    
    def _print_final_stats(self, elapsed_time: float):
        """최종 통계 출력"""
        stats = self.scheduler.get_stats()
        
        print("\n" + "=" * 60)
        print("ELIS Experiment Results")
        print("=" * 60)
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Completed jobs: {stats['completed_jobs']}")
        print(f"Average JCT: {stats['avg_jct']:.2f}s")
        print(f"Total preemptions: {stats['total_preemptions']}")
        print(f"Predictions made: {stats['predictions_made']}")
        print("=" * 60)
        
        # 결과에 통계 추가
        self.results["final_stats"] = stats
        self.results["elapsed_time"] = elapsed_time
    
    def save_results(self, output_path: str):
        """결과 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[Saved] Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ELIS: Efficient LLM Iterative Scheduling System"
    )
    
    # 모드 선택
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "live"],
        default="simulation",
        help="Execution mode (default: simulation)"
    )
    
    # vLLM 설정
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1/completions",
        help="vLLM server URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name"
    )
    
    # 스케줄링 설정
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Maximum batch size"
    )
    parser.add_argument(
        "--prediction-interval",
        type=int,
        default=50,
        help="Re-prediction interval (tokens)"
    )
    
    # Gamma distribution 설정
    parser.add_argument(
        "--gamma-shape",
        type=float,
        default=1.0,
        help="Gamma distribution shape parameter"
    )
    parser.add_argument(
        "--gamma-scale",
        type=float,
        default=1.0,
        help="Gamma distribution scale parameter"
    )
    
    # 기타 설정
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./train/checkpoints/latest_model.pt",
        help="Predictor model checkpoint path"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="./data/processed_dataset.json",
        help="Prompts JSON file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./scheduler/results/elis_results.json",
        help="Output results file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Config 생성
    config = ELISConfig(
        vllm_server_url=args.vllm_url,
        model_name=args.model,
        predictor_checkpoint=args.checkpoint,
        prediction_interval=args.prediction_interval,
        max_batch_size=args.batch_size,
        gamma_shape=args.gamma_shape,
        gamma_scale=args.gamma_scale,
        prompts_file=args.prompts_file,
        total_requests=args.num_requests,
        seed=args.seed,
        device=args.device,
    )
    
    # Runner 생성 및 실행
    runner = ELISRunner(config)
    
    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "simulation":
        runner.run_simulation(args.num_requests)
    else:
        runner.run_live(args.num_requests)
    
    # 결과 저장
    runner.save_results(args.output)


if __name__ == "__main__":
    main()

