import re
import time
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_FILE = "huggingface_api_key.txt"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"  # Corrected URL

def load_api_key():
    try:
        with open(API_KEY_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise Exception(f"API key file {API_KEY_FILE} not found")

HEADERS = {
    "Authorization": f"Bearer {load_api_key()}",
    "Content-Type": "application/json"
}

def query_inference_api(payload):
    retries = 3
    for _ in range(retries):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:  # Model not loaded
            result = response.json()
            wait_time = result.get('estimated_time', 10)
            logger.warning(f"Model loading, retrying in {wait_time:.1f}s")
            time.sleep(wait_time)
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")
    
    raise Exception("Failed to get valid response after retries")

def extract_answers(raw_answer, expected_order):
    answer_dict = {}
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
            answer_dict[category] = "NO"
    
    return [answer_dict[cat] for cat in expected_order]

def analyze_script(script):
    logger.info("Starting Analysis")
    start_time = time.time()
    
    try:
        max_chunk_size = 2048
        chunks = [script[i:i+max_chunk_size] for i in range(0, len(script), max_chunk_size)]
        
        expected_order = [
            "VIOLENCE", "DEATH", "SUBSTANCE_USE", "GORE", 
            "VOMIT", "SEXUAL_CONTENT", "SEXUAL_ABUSE", 
            "SELF_HARM", "GUN_USE", "ANIMAL_CRUELTY", "MENTAL_HEALTH"
        ]
        
        identified = {cat: 0 for cat in expected_order}
        
        for chunk_idx, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {chunk_idx}/{len(chunks)}")
            
            prompt = f"""Comprehensive Sensitive Content Analysis Protocol

[Your original prompt here...]
{chunk}

RESPONSE FORMAT:
[Your original format here...]"""

            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_new_tokens": 1024,
                    "return_full_text": False
                }
            }
            
            result = query_inference_api(payload)
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                raw_answer = result[0].get('generated_text', '')
            else:
                raw_answer = result.get('generated_text', '')
            
            logger.info("[Model Raw Response]")
            logger.info(raw_answer)
            
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
        logger.error(f"Analysis error: {e}")
        return {"error": str(e)}

def get_detailed_analysis(script):
    return analyze_script(script)