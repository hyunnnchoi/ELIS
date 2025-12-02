"""
ELIS 예측 예제 스크립트

latest_model.pt를 로드하여 샘플 텍스트에 대한 remaining token 예측을 수행합니다.
"""

import torch
from transformers import AutoTokenizer
import sys
import os

# train 모듈 import를 위한 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))
from model import ELISPredictor


def load_model(checkpoint_path, device='cuda'):
    """체크포인트에서 모델 로드"""
    print(f"Loading model from {checkpoint_path}...")
    
    # 모델 초기화
    model = ELISPredictor(
        bge_model_name="BAAI/bge-base-en-v1.5",
        hidden_dim=1024,
        num_layers=8,
        freeze_bge=True
    )
    
    # [NOTE, hyunnnchoi, 2025.12.01] 체크포인트 로드 (weights_only=False 설정)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    else:
        print(f"  Best val loss: N/A")
    
    return model


def predict(model, tokenizer, text, device='cuda', max_length=512):
    """텍스트에 대한 remaining token 예측"""
    # 토크나이즈
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # 예측
    with torch.no_grad():
        prediction = model(input_ids, attention_mask)
    
    return prediction.item()


def main():
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # 모델 및 토크나이저 로드
    checkpoint_path = '/home/work/hyunmokchoi/ELIS/train/checkpoints/latest_model.pt'
    model = load_model(checkpoint_path, device=device)
    
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    print()
    
    # 테스트 샘플들
    test_samples = [
        {
            "prompt": "What is the capital of France?",
            "generated": "",
            "description": "간단한 질문 (답변 시작 전)"
        },
        {
            "prompt": "What is the capital of France?",
            "generated": "The capital of France is",
            "description": "간단한 질문 (일부 답변 생성)"
        },
        {
            "prompt": "Explain quantum mechanics in detail.",
            "generated": "",
            "description": "복잡한 질문 (답변 시작 전)"
        },
        {
            "prompt": "Explain quantum mechanics in detail.",
            "generated": "Quantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at the atomic and subatomic levels.",
            "description": "복잡한 질문 (일부 답변 생성)"
        },
        {
            "prompt": "Write a short story about a robot.",
            "generated": "",
            "description": "창작 요청 (답변 시작 전)"
        },
        {
            "prompt": "Write a short story about a robot.",
            "generated": "Once upon a time, in a bustling city of steel and glass, there lived a small robot named Bolt. Bolt was different from other robots - he had a curious mind and a kind heart.",
            "description": "창작 요청 (일부 답변 생성)"
        }
    ]
    
    # 예측 실행
    print("=" * 80)
    print("ELIS Output Token Predictor - Prediction Results")
    print("=" * 80)
    print()
    
    for i, sample in enumerate(test_samples, 1):
        # 전체 컨텍스트 생성 (프롬프트 + 현재까지 생성된 텍스트)
        if sample['generated']:
            full_context = sample['prompt'] + " " + sample['generated']
        else:
            full_context = sample['prompt']
        
        # 예측
        remaining_tokens = predict(model, tokenizer, full_context, device)
        
        # 결과 출력
        print(f"Sample {i}: {sample['description']}")
        print(f"  Prompt: {sample['prompt']}")
        if sample['generated']:
            print(f"  Generated so far: {sample['generated'][:100]}...")
        else:
            print(f"  Generated so far: (none)")
        print(f"  Predicted remaining tokens: {remaining_tokens:.2f}")
        print()
    
    print("=" * 80)
    print("\n예측 완료! 모델이 각 상황에서 앞으로 생성될 토큰의 개수를 예측했습니다.")
    print("일반적으로 복잡한 질문일수록, 그리고 답변이 덜 진행되었을수록 remaining token이 많습니다.")


if __name__ == "__main__":
    main()

