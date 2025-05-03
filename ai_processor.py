import os
import json
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_api_key():
    """
    Get API key from environment variables
    """
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        logger.warning("TOGETHER_API_KEY environment variable not set")
        raise ValueError("TOGETHER_API_KEY environment variable not set. Please set this to use AI features.")
    return api_key

def call_together_api(prompt, max_tokens=800):
    """
    Call the Together.ai API with the given prompt
    """
    api_key = get_api_key()
    api_url = "https://api.together.xyz/v1/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3-8b-instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "stop": ["<|im_end|>", "</answer>"]
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            generated_text = result['choices'][0]['text'].strip()
            return generated_text
        else:
            logger.error(f"Unexpected API response format: {result}")
            raise ValueError("Unexpected API response format")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise ValueError(f"API request failed: {str(e)}")

def get_improvement_suggestions(resume_text, job_description):
    """
    Use Together.ai's API with LLaMA 3 to suggest improvements
    """
    # Create the prompt
    prompt = f"""<|im_start|>system
You are an expert ATS (Applicant Tracking System) consultant who helps job seekers improve their resumes to better match job descriptions. Your task is to analyze the resume against the job description and provide 3-5 specific, actionable suggestions to improve the ATS match score.

Focus on:
1. Keyword alignment
2. Skills and qualifications
3. Experience relevance
4. Achievements that matter for this role
5. Professional formatting

Provide clear, bulleted suggestions that the applicant can immediately implement.
<|im_end|>

<|im_start|>user
REVIEW THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Please provide 3-5 specific, actionable suggestions to improve this resume's ATS match score for the job description above.
<|im_end|>

<|im_start|>assistant
<answer>
"""
    
    try:
        suggestions = call_together_api(prompt, max_tokens=800)
        return suggestions
    except Exception as e:
        logger.error(f"Error getting improvement suggestions: {str(e)}")
        return "Unable to generate suggestions at this time. Please try again later."

def rewrite_resume(resume_text, job_description):
    """
    Use Together.ai's API with LLaMA 3 to rewrite the resume
    """
    # Create the prompt
    prompt = f"""<|im_start|>system
You are an expert resume writer who specializes in optimizing resumes to pass ATS (Applicant Tracking System) scans. Your task is to rewrite the provided resume to better match the job description, focusing on improving keyword alignment, highlighting relevant experience, and emphasizing transferable skills.

Follow these guidelines:
1. Maintain the same section structure as the original resume
2. Preserve all contact information and personal details
3. Highlight relevant skills and experience that match the job description
4. Incorporate keywords from the job description naturally
5. Quantify achievements where possible
6. Keep a professional tone
7. Maintain the same general length as the original

Do NOT invent work experience or qualifications that aren't mentioned in the original resume.
<|im_end|>

<|im_start|>user
REWRITE THE FOLLOWING RESUME TO BETTER MATCH THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME:
{resume_text}

Please rewrite this resume to maximize its ATS match score while maintaining the person's honest work history and qualifications.
<|im_end|>

<|im_start|>assistant
<answer>
"""
    
    try:
        rewritten_resume = call_together_api(prompt, max_tokens=1500)
        return rewritten_resume
    except Exception as e:
        logger.error(f"Error rewriting resume: {str(e)}")
        return "Unable to rewrite resume at this time. Please try again later."
