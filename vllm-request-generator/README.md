# vLLM 서버에 프롬프트 전송하기

이 스크립트는 `processed_dataset.json` 파일에 있는 프롬프트들을 vLLM 서버에 전송하고 결과를 저장합니다.

## 설치

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

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
  --max-tokens 512 \
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

