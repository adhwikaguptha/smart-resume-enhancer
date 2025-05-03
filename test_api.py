import os
import requests
import logging
import sys

# Configure logging to print to console
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_hf_api_token():
    """
    Get Hugging Face API token from environment variables
    """
    api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    if not api_token:
        logger.warning("HUGGINGFACEHUB_API_TOKEN environment variable not set")
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set. Please set this to use AI features.")
    return api_token

def test_hf_api():
    """
    Test the Hugging Face API with a simple prompt
    """
    api_token = get_hf_api_token()
    api_url = "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Simple prompt for testing
    data = {
        "inputs": "Hello, how are you today?",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    logger.info(f"Testing API request to: {api_url}")
    logger.info(f"Headers: {headers}")
    logger.info(f"Data: {data}")
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        try:
            # Try to parse as JSON first
            result = response.json()
            logger.info(f"Response JSON: {result}")
            
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and 'generated_text' in result[0]:
                logger.info(f"Generated text: {result[0]['generated_text']}")
            elif isinstance(result, dict) and 'generated_text' in result:
                logger.info(f"Generated text: {result['generated_text']}")
            else:
                logger.info("Unexpected response format")
        except Exception as e:
            # If not JSON, print the raw text
            logger.error(f"Failed to parse response as JSON: {e}")
            logger.info(f"Raw response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")

if __name__ == "__main__":
    test_hf_api()
