#!/usr/bin/env python3
"""
vLLM 서버에 프롬프트를 전송하는 스크립트
"""
import json
import requests
from typing import List, Dict, Optional
import time
from tqdm import tqdm
# [NOTE, hyunnnchoi, 2025.11.12] tokenizer 추가 for output slicing
from transformers import AutoTokenizer
# [NOTE, hyunnnchoi, 2025.11.12] 병렬 처리를 위한 모듈 추가
from concurrent.futures import ThreadPoolExecutor, as_completed

# [NOTE, hyunnnchoi, 2025.11.12] vLLM 서버 설정
VLLM_SERVER_URL = "http://localhost:8000/v1/completions"  # vLLM 서버 URL
# [NOTE, hyunnnchoi, 2025.11.12] 모델 이름을 gpt-oss-20b로 변경
MODEL_NAME = "gpt-oss-20b"  # 사용할 모델 이름


def load_prompts(json_file_path: str) -> List[str]:
    """
    JSON 파일에서 프롬프트 목록을 로드합니다.
    
    Args:
        json_file_path: JSON 파일 경로
        
    Returns:
        프롬프트 문자열 리스트
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('prompts', [])


# [NOTE, hyunnnchoi, 2025.11.12] 50토큰씩 누적으로 output을 자르는 함수 추가
def slice_output_by_tokens(output_text: str, tokenizer, chunk_size: int = 50) -> List[Dict]:
    """
    output 텍스트를 50토큰씩 누적으로 자르고, 각 청크에 대한 정보를 반환합니다.
    
    Args:
        output_text: 전체 output 텍스트
        tokenizer: 사용할 tokenizer
        chunk_size: 청크 크기 (기본값: 50)
        
    Returns:
        각 청크의 정보를 담은 딕셔너리 리스트
        [
            {"output_text": "0~50토큰", "num_tokens": 50, "remaining_tokens": 100},
            {"output_text": "0~100토큰", "num_tokens": 100, "remaining_tokens": 50},
            ...
        ]
    """
    # output 텍스트를 토큰화
    tokens = tokenizer.encode(output_text, add_special_tokens=False)
    total_tokens = len(tokens)
    
    chunks = []
    current_pos = 0
    
    # 50토큰씩 누적으로 자르기
    while current_pos < total_tokens:
        next_pos = min(current_pos + chunk_size, total_tokens)
        
        # 0부터 next_pos까지의 토큰을 디코딩
        chunk_tokens = tokens[:next_pos]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        chunks.append({
            "output_text": chunk_text,
            "num_tokens": next_pos,
            "remaining_tokens": total_tokens - next_pos
        })
        
        current_pos = next_pos
    
    # 마지막 청크가 정확히 total_tokens가 아니면 전체를 추가
    if not chunks or chunks[-1]["num_tokens"] < total_tokens:
        full_text = tokenizer.decode(tokens, skip_special_tokens=True)
        chunks.append({
            "output_text": full_text,
            "num_tokens": total_tokens,
            "remaining_tokens": 0
        })
    
    return chunks


def send_to_vllm(
    prompt: str,
    server_url: str = VLLM_SERVER_URL,
    model_name: str = MODEL_NAME,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    timeout: int = 300
) -> Optional[Dict]:
    """
    단일 프롬프트를 vLLM 서버에 전송합니다.
    
    Args:
        prompt: 전송할 프롬프트
        server_url: vLLM 서버 URL
        model_name: 모델 이름
        max_tokens: 생성할 최대 토큰 수 (None이면 제한 없음)
        temperature: 샘플링 온도
        top_p: nucleus sampling 파라미터
        timeout: 요청 타임아웃 (초)
        
    Returns:
        API 응답 또는 None (에러 발생 시)
    """
    # [NOTE, hyunnnchoi, 2025.11.12] max_tokens 제한 제거 옵션 추가
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    # max_tokens가 지정된 경우만 추가
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    
    try:
        response = requests.post(
            server_url, 
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 실패: {str(e)}")
        return None


def process_all_prompts(
    json_file_path: str,
    output_file_path: str,
    server_url: str = VLLM_SERVER_URL,
    model_name: str = MODEL_NAME,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    batch_delay: float = 0.0,
    tokenizer_path: str = None,
    batch_size: int = 16
):
    """
    모든 프롬프트를 처리하고 결과를 저장합니다.
    
    Args:
        json_file_path: 입력 JSON 파일 경로
        output_file_path: 출력 JSONL 파일 경로
        server_url: vLLM 서버 URL
        model_name: 모델 이름
        max_tokens: 생성할 최대 토큰 수 (None이면 제한 없음)
        temperature: 샘플링 온도
        batch_delay: 각 요청 사이의 대기 시간 (초)
        tokenizer_path: tokenizer 경로 (None이면 model_name 사용)
        batch_size: 동시에 처리할 요청 수
    """
    # [NOTE, hyunnnchoi, 2025.11.12] tokenizer 로드
    print(f"🔧 Tokenizer 로딩 중...")
    if tokenizer_path is None:
        tokenizer_path = "/model"  # vLLM 서버의 모델 경로
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"✅ Tokenizer 로드 완료\n")
    except Exception as e:
        print(f"⚠️ Tokenizer 로드 실패, 기본 tokenizer 사용: {e}\n")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # fallback
    
    print(f"📂 파일 로딩 중: {json_file_path}")
    prompts = load_prompts(json_file_path)
    print(f"✅ {len(prompts)}개의 프롬프트를 로드했습니다.\n")
    
    results = []
    training_data = []  # JSONL 형식으로 저장할 학습 데이터
    success_count = 0
    fail_count = 0
    total_training_samples = 0
    
    print(f"🚀 vLLM 서버로 요청 전송 시작...")
    print(f"   서버 URL: {server_url}")
    print(f"   모델: {model_name}")
    print(f"   배치 크기: {batch_size}")
    print(f"   Max tokens: {'제한 없음' if max_tokens is None else max_tokens}\n")
    
    # [NOTE, hyunnnchoi, 2025.11.12] 병렬 처리를 위한 헬퍼 함수
    def process_single_prompt(idx_prompt_tuple):
        idx, prompt = idx_prompt_tuple
        response = send_to_vllm(
            prompt=prompt,
            server_url=server_url,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return idx, prompt, response
    
    # [NOTE, hyunnnchoi, 2025.11.12] ThreadPoolExecutor로 병렬 처리
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # 모든 프롬프트를 인덱스와 함께 제출
        futures = {
            executor.submit(process_single_prompt, (idx, prompt)): idx
            for idx, prompt in enumerate(prompts)
        }
        
        # 완료된 작업을 처리
        for future in tqdm(as_completed(futures), total=len(prompts), desc="처리 중"):
            try:
                idx, prompt, response = future.result()
                
                # input/output을 명확하게 저장하고 50토큰씩 누적으로 자르기
                if response:
                    # 생성된 텍스트 추출
                    output_text = ""
                    if "choices" in response and len(response["choices"]) > 0:
                        output_text = response["choices"][0].get("text", "")
                    
                    # output을 50토큰씩 누적으로 자르기
                    if output_text.strip():
                        chunks = slice_output_by_tokens(output_text, tokenizer, chunk_size=50)
                        
                        # 각 청크를 training_data에 추가
                        for chunk in chunks:
                            training_entry = {
                                "input_prompt": prompt,
                                "output_prompt": chunk["output_text"],
                                "number_of_output_tokens": chunk["num_tokens"],
                                "remaining_tokens": chunk["remaining_tokens"]
                            }
                            training_data.append(training_entry)
                            total_training_samples += 1
                    
                    result_entry = {
                        "index": idx,
                        "input_prompt": prompt,
                        "output_text": output_text,
                        "full_response": response,
                        "status": "success"
                    }
                    success_count += 1
                else:
                    result_entry = {
                        "index": idx,
                        "input_prompt": prompt,
                        "output_text": "",
                        "error": "Request failed",
                        "status": "failed"
                    }
                    fail_count += 1
                
                results.append(result_entry)
                
            except Exception as e:
                print(f"\n❌ 예외 발생 (idx={futures[future]}): {str(e)}")
                result_entry = {
                    "index": futures[future],
                    "input_prompt": prompts[futures[future]],
                    "output_text": "",
                    "error": str(e),
                    "status": "failed"
                }
                results.append(result_entry)
                fail_count += 1
    
    # [NOTE, hyunnnchoi, 2025.11.12] JSONL 형식으로 학습 데이터 저장
    training_output_path = output_file_path.replace('.json', '_training.jsonl')
    print(f"\n💾 학습 데이터 저장 중: {training_output_path}")
    with open(training_output_path, 'w', encoding='utf-8') as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 원본 결과도 JSON으로 저장 (디버깅용)
    print(f"💾 원본 결과 저장 중: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(prompts),
            "success": success_count,
            "failed": fail_count,
            "total_training_samples": total_training_samples,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 완료!")
    print(f"   총 프롬프트: {len(prompts)}개")
    print(f"   성공: {success_count}, 실패: {fail_count}")
    print(f"   학습 데이터 샘플 수: {total_training_samples}개")
    print(f"   학습 데이터 파일: {training_output_path}")
    print(f"   원본 결과 파일: {output_file_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM 서버에 프롬프트 전송")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/processed_dataset.json",
        help="입력 JSON 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/vllm_results.json",
        help="출력 JSON 파일 경로"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=VLLM_SERVER_URL,
        help="vLLM 서버 URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="모델 이름"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="생성할 최대 토큰 수 (기본값: None, 제한 없음)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="샘플링 온도"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="각 요청 사이의 대기 시간 (초)"
    )
    # [NOTE, hyunnnchoi, 2025.11.12] tokenizer 경로 인자 추가
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer 경로 (기본값: /model)"
    )
    # [NOTE, hyunnnchoi, 2025.11.12] 배치 크기 인자 추가
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="동시에 처리할 요청 수 (기본값: 16)"
    )
    
    args = parser.parse_args()
    
    process_all_prompts(
        json_file_path=args.input,
        output_file_path=args.output,
        server_url=args.server_url,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_delay=args.delay,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size
    )

