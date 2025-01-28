import re
import time
import logging
import requests
from typing import Dict, List, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_FILE = "huggingface_api_key.txt"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

def load_api_key() -> str:
    try:
        with open(API_KEY_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise APIError(f"API key file {API_KEY_FILE} not found")

HEADERS = {
    "Authorization": f"Bearer {load_api_key()}",
    "Content-Type": "application/json"
}

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((requests.exceptions.RequestException, APIError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Attempt {retry_state.attempt_number} failed, retrying in {retry_state.next_action.sleep} seconds..."
    )
)
def query_inference_api(payload: Dict) -> Dict:
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=90)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:  # Model not loaded
            result = response.json()
            wait_time = min(result.get('estimated_time', 10), 30)  # Cap wait time at 30s
            logger.warning(f"Model loading, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            raise APIError("Model still loading")
        else:
            raise APIError(f"API Error {response.status_code}: {response.text}")
    
    except requests.exceptions.Timeout:
        raise APIError("Request timed out - model may be too busy")
    except requests.exceptions.RequestException as e:
        raise APIError(f"Request failed: {str(e)}")

def extract_answers(raw_answer: str, expected_order: List[str]) -> List[str]:
    answer_dict = {}
    
    # Convert raw answer to uppercase for consistent matching
    raw_answer = raw_answer.upper()
    
    for category in expected_order:
        patterns = [
            rf"{category}[\s:]*\[?(YES|NO|MAYBE|Y|N|M)\]?",
            rf"{category.replace('_', ' ')}[\s:]*\[?(YES|NO|MAYBE|Y|N|M)\]?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_answer, re.IGNORECASE)
            if match:
                raw_result = match.group(1).upper()
                answer = "YES" if raw_result in ("Y", "YES") else "NO" if raw_result in ("N", "NO") else "MAYBE"
                answer_dict[category] = answer
                break
        else:
            answer_dict[category] = "NO"  # Default to NO if no match found
    
    return [answer_dict[cat] for cat in expected_order]

def analyze_script(script: str) -> Dict[str, Union[int, str]]:
    logger.info("Starting Analysis")
    start_time = time.time()
    
    try:
        # Reduced chunk size to help with timeouts
        max_chunk_size = 1024
        chunks = [script[i:i+max_chunk_size] for i in range(0, len(script), max_chunk_size)]
        
        expected_order = [
            "VIOLENCE", "DEATH", "SUBSTANCE_USE", "GORE", 
            "VOMIT", "SEXUAL_CONTENT", "SEXUAL_ABUSE", 
            "SELF_HARM", "GUN_USE", "ANIMAL_CRUELTY", "MENTAL_HEALTH"
        ]
        
        identified = {cat: 0 for cat in expected_order}
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {chunk_idx}/{len(chunks)}")
            
            prompt = f"""Analyze the following text for sensitive content, responding only with category labels and YES/NO/MAYBE:

{chunk}

Format: CATEGORY: YES/NO/MAYBE"""

            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.1,  # Reduced temperature for more consistent results
                    "top_p": 0.9,
                    "max_new_tokens": 512,  # Reduced token limit
                    "return_full_text": False
                }
            }
            
            result = query_inference_api(payload)
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                raw_answer = result[0].get('generated_text', '')
            else:
                raw_answer = result.get('generated_text', '')
            
            logger.debug("[Model Raw Response]")
            logger.debug(raw_answer)
            
            answers = extract_answers(raw_answer, expected_order)
            
            logger.info("[Analysis Results]")
            for cat, ans in zip(expected_order, answers):
                logger.info(f"{cat}: {ans}")
                if ans == "YES":
                    identified[cat] += 1
        
        logger.info("\n=== Final Results ===")
        for cat in expected_order:
            score = identified[cat]
            status = "CONFIRMED" if score > 0 else "NOT FOUND"
            logger.info(f"{cat}: {status} ({score}/{len(chunks)} chunks)")
        
        total_time = time.time() - start_time
        logger.info(f"Total analysis time: {total_time:.1f}s")
        return identified
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"error": str(e)}

def get_detailed_analysis(script: str) -> Dict[str, Union[int, str]]:
    return analyze_script(script)