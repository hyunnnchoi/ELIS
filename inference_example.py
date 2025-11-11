"""
Example script for sending processed prompts to vLLM server.
This can be used both from host and inside container.
"""
import requests
import json
from typing import List, Dict, Optional
from dataset import get_prompts
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for interacting with vLLM OpenAI-compatible API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize vLLM client.
        
        Args:
            base_url: Base URL of vLLM server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.completions_url = f"{self.base_url}/v1/completions"
        
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Optional[Dict]:
        """
        Send a prompt to vLLM and get response.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters for vLLM API
            
        Returns:
            Response dictionary or None if error
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        try:
            response = requests.post(
                self.completions_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling vLLM API: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


def run_inference_batch(
    prompts: List[str],
    vllm_url: str = "http://localhost:8000",
    batch_size: int = 10,
    output_file: str = "./data/inference_results.jsonl"
):
    """
    Run batch inference on all prompts.
    
    Args:
        prompts: List of prompts to process
        vllm_url: vLLM server URL
        batch_size: Number of prompts to process before logging progress
        output_file: File to save results
    """
    client = VLLMClient(base_url=vllm_url)
    
    # Check server health
    logger.info("Checking vLLM server health...")
    if not client.health_check():
        logger.error(f"vLLM server at {vllm_url} is not responding!")
        return
    
    logger.info(f"vLLM server is healthy. Starting inference on {len(prompts)} prompts...")
    
    results = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, prompt in enumerate(prompts, 1):
            # Generate response
            response = client.generate(prompt)
            
            if response:
                result = {
                    "prompt_id": idx,
                    "prompt": prompt,
                    "response": response
                }
                results.append(result)
                
                # Write to JSONL file
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                
                # Log progress
                if idx % batch_size == 0:
                    logger.info(f"Processed {idx}/{len(prompts)} prompts")
            else:
                logger.warning(f"Failed to process prompt {idx}")
    
    logger.info(f"Inference complete! Results saved to {output_file}")
    logger.info(f"Successfully processed {len(results)}/{len(prompts)} prompts")


def main():
    """Main function for running inference."""
    
    # Load processed prompts
    logger.info("Loading processed dataset...")
    prompts = get_prompts()
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Run inference
    # Note: Change vllm_url if vLLM is running on different host/port
    # For container: use service name from docker-compose (e.g., http://vllm:8000)
    # For host: use localhost
    vllm_url = "http://localhost:8000"  # Change this as needed
    
    logger.info(f"Starting inference with vLLM at {vllm_url}")
    
    # You can process a subset for testing
    # test_prompts = prompts[:100]  # Process first 100 prompts
    test_prompts = prompts  # Process all prompts
    
    run_inference_batch(
        prompts=test_prompts,
        vllm_url=vllm_url,
        batch_size=10
    )


if __name__ == "__main__":
    main()

