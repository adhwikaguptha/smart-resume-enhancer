import os
import json
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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

def call_groq_api(prompt, max_tokens=800):
    """
    Call the Groq API with the given prompt
    Using the llama2-70b-4096 model
    """
    api_key = get_groq_api_key()
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Format for the Groq chat completions API
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful resume optimization assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    logger.info(f"API request to: {api_url}")
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        logger.info(f"API response status: {response.status_code}")
        
        # Groq API returns in OpenAI format
        result = response.json()
        logger.debug(f"API response: {result}")
        
        if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
            generated_text = result['choices'][0]['message']['content'].strip()
            return generated_text
        else:
            logger.error(f"Unexpected API response format: {result}")
            raise ValueError("Unexpected API response format")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise ValueError(f"API request failed: {str(e)}")

def get_improvement_suggestions(resume_text, job_description):
    """
    Use Groq API to get improvement suggestions
    """
    # Create the prompt for the model
    prompt = f"""REVIEW THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Provide 4-6 specific, actionable changes that would make this resume more effective at passing ATS screening for this job. For each suggestion:
1. Clearly identify what specific aspect of the resume should be changed
2. Provide a concrete recommendation for the improvement
3. Explain why this change will increase the ATS score

Format your suggestions with bullet points, and be sure to include examples of keywords from the job description that should be incorporated."""

    
    try:
        suggestions = call_groq_api(prompt, max_tokens=800)
        return suggestions
    except Exception as e:
        logger.error(f"Error getting improvement suggestions: {str(e)}")
        return "Unable to generate suggestions at this time. Please try again later."

def rewrite_resume(resume_text, job_description):
    """
    Use Groq API to rewrite the resume
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

Provide ONLY the rewritten resume text, without any explanation of changes or formatting notes. Focus exclusively on delivering a clean, optimized resume document ready for immediate use.

Do NOT invent work experience or qualifications that aren't mentioned in the original resume."""

    
    try:
        rewritten_resume = call_groq_api(prompt, max_tokens=1500)
        return rewritten_resume
    except Exception as e:
        logger.error(f"Error rewriting resume: {str(e)}")
        return "Unable to rewrite resume at this time. Please try again later."
