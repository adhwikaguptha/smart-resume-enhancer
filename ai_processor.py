import os
import json
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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

def call_hf_api(prompt, max_new_tokens=800):
    """
    Call the Hugging Face Hub API with the given prompt
    Using the TinyLlama/TinyLlama-1.1B-Chat-v1.0 model (free to use)
    """
    api_token = get_hf_api_token()
    api_url = "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Format for the Hugging Face inference API
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    logger.info(f"API request to: {api_url}")
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        logger.info(f"API response status: {response.status_code}")
        
        # Hugging Face API typically returns a list with a single text element
        result = response.json()
        logger.debug(f"API response: {result}")
        
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and 'generated_text' in result[0]:
            # Standard inference API response format
            generated_text = result[0]['generated_text']
            # Remove the prompt from the response if it's included
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            return generated_text.strip()
        elif isinstance(result, dict) and 'generated_text' in result:
            # Alternative response format
            generated_text = result['generated_text']
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            return generated_text.strip()
        else:
            logger.error(f"Unexpected API response format: {result}")
            raise ValueError("Unexpected API response format")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise ValueError(f"API request failed: {str(e)}")

def get_improvement_suggestions(resume_text, job_description):
    """
    Use Hugging Face Hub API to get improvement suggestions
    """
    # Create the prompt for the model
    prompt = f"""REVIEW THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Please provide 3-5 specific, actionable suggestions to improve this resume's ATS match score for the job description above. Format with bullet points."""

    
    try:
        suggestions = call_hf_api(prompt, max_new_tokens=800)
        return suggestions
    except Exception as e:
        logger.error(f"Error getting improvement suggestions: {str(e)}")
        return "Unable to generate suggestions at this time. Please try again later."

def rewrite_resume(resume_text, job_description):
    """
    Use Hugging Face Hub API to rewrite the resume
    """
    # Create the prompt for the model
    prompt = f"""You are an expert resume writer who specializes in optimizing resumes to pass ATS (Applicant Tracking System) scans.
    
REWRITE THE FOLLOWING RESUME TO BETTER MATCH THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME:
{resume_text}

Please rewrite this resume to maximize its ATS match score while maintaining the person's honest work history and qualifications. Follow these guidelines:
1. Maintain the same section structure as the original resume
2. Preserve all contact information and personal details
3. Highlight relevant skills and experience that match the job description
4. Incorporate keywords from the job description naturally
5. Quantify achievements where possible
6. Keep a professional tone
7. Maintain the same general length as the original

Do NOT invent work experience or qualifications that aren't mentioned in the original resume."""

    
    try:
        rewritten_resume = call_hf_api(prompt, max_new_tokens=1500)
        return rewritten_resume
    except Exception as e:
        logger.error(f"Error rewriting resume: {str(e)}")
        return "Unable to rewrite resume at this time. Please try again later."
