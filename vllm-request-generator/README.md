# vLLM 서버에 프롬프트 전송하기

이 스크립트는 `processed_dataset.json` 파일에 있는 프롬프트들을 vLLM 서버에 전송하고 결과를 저장합니다.

## 설치

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

## 데이터 경로

- **입력 데이터셋**: vLLM 컨테이너 루트 디렉토리의 `/data` 폴더에서 읽어옵니다.
- **출력 데이터셋**: vLLM 컨테이너 루트 디렉토리의 `/data/{파일명}.jsonl` 형식으로 저장됩니다.

> **Note**: 이 스크립트는 vLLM 서버로부터 응답을 받아 임베딩 모델 학습용 데이터셋을 생성합니다.

## 사용 방법

### 1. 기본 사용법

```bash
python send_to_vllm.py
```

### 2. 커스텀 설정

```bash
python send_to_vllm.py \
  --input /data/processed_dataset.json \
  --output /data/vllm_results.json \
  --server-url http://localhost:8000/v1/completions \
  --model your-model-name \
  --max-tokens 1024 \
  --temperature 0.7 \
  --delay 0.1
```

### 3. 파라미터 설명

- `--input`: 입력 JSON 파일 경로 (기본값: `/data/processed_dataset.json`)
- `--output`: 출력 JSON 파일 경로 (기본값: `/data/vllm_results.json`)
- `--server-url`: vLLM 서버 URL (기본값: `http://localhost:8000/v1/completions`)
- `--model`: 사용할 모델 이름 (기본값: `your-model-name`)
- `--max-tokens`: 생성할 최대 토큰 수 (기본값: 512)
- `--temperature`: 샘플링 온도 (기본값: 0.7)
- `--delay`: 각 요청 사이의 대기 시간, 초 단위 (기본값: 0.0)

## 사용 예시

### 예시 1: 로컬 vLLM 서버에 연결

```bash
python send_to_vllm.py \
  --server-url http://localhost:8000/v1/completions \
  --model llama-2-7b-chat
```

### 예시 2: 원격 서버에 연결하고 요청 간 지연 추가

```bash
python send_to_vllm.py \
  --server-url http://your-server:8000/v1/completions \
  --model mistral-7b-instruct \
  --delay 0.5
```

### 예시 3: 더 긴 응답 생성

```bash
python send_to_vllm.py \
  --max-tokens 2048 \
  --temperature 0.9
```

### 예시 4: 백그라운드 실행 (프로덕션 환경)

```bash
cd /vllm/kukt && nohup python3 send_to_vllm.py \
  --input /data/processed_dataset.json \
  --output /data/vllm_results.json \
  --model gpt-oss-20b \
  --batch-size 16 \
  --max-tokens 1024 \
  > /data/vllm_run.log 2>&1 &

echo "✅ 백그라운드에서 실행 시작 (max_tokens=1024, 실시간 저장)"
echo "   PID: $!"
echo "   로그: tail -f /data/vllm_run.log"
echo "   학습 데이터: watch -n 5 'wc -l /data/vllm_results_training.jsonl'"
```

이 예시는 다음과 같은 기능을 제공합니다:
- `nohup`을 사용하여 백그라운드에서 안전하게 실행
- 로그를 `/data/vllm_run.log`에 실시간으로 저장
- 배치 크기 16으로 효율적인 처리
- 실행 후 프로세스 ID와 모니터링 명령어 표시

## 출력 형식

결과는 다음과 같은 JSON 형식으로 저장됩니다:

```json
{
  "total": 100,
  "success": 98,
  "failed": 2,
  "results": [
    {
      "index": 0,
      "prompt": "Your prompt here...",
      "response": {
        "id": "cmpl-xxx",
        "object": "text_completion",
        "created": 1234567890,
        "model": "your-model-name",
        "choices": [
          {
            "text": "Generated text...",
            "index": 0,
            "logprobs": null,
            "finish_reason": "length"
          }
        ]
      },
      "status": "success"
    }
  ]
}
```

## 주의사항

1. vLLM 서버가 실행 중이어야 합니다.
2. 서버 URL과 모델 이름을 올바르게 설정해야 합니다.
3. 대량의 요청을 보낼 경우 `--delay` 옵션으로 서버 부하를 조절할 수 있습니다.

