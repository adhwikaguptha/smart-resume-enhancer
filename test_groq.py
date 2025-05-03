import os
import requests
import logging
import sys

# Configure logging to print to console
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_groq_api_key():
    """
    Get Groq API key from environment variables
    """
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        logger.warning("GROQ_API_KEY environment variable not set")
        raise ValueError("GROQ_API_KEY environment variable not set. Please set this to use AI features.")
    return api_key

def test_groq_api():
    """
    Test the Groq API with a simple prompt
    """
    api_key = get_groq_api_key()
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple prompt for testing
    data = {
        "model": "llama2-70b-4096",
        "messages": [
            {"role": "system", "content": "You are a helpful resume optimization assistant."},
            {"role": "user", "content": "Write a brief sentence about why ATS optimization is important."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
    
    logger.info(f"Testing API request to: {api_url}")
    logger.info(f"Headers: {headers}")
    logger.info(f"Data: {data}")
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        try:
            # Try to parse as JSON
            result = response.json()
            logger.info(f"Response JSON: {result}")
            
            if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                logger.info(f"Generated text: {result['choices'][0]['message']['content']}")
            else:
                logger.info("Unexpected response format")
        except Exception as e:
            # If not JSON, print the raw text
            logger.error(f"Failed to parse response as JSON: {e}")
            logger.info(f"Raw response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")

if __name__ == "__main__":
    test_groq_api()
