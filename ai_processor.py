import os
import json
import requests
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def call_groq_api(prompt, system_prompt="You are a helpful resume optimization assistant.", max_tokens=800, temperature=0.2):
    """
    Call the Groq API with the given prompt
    Using Groq's LLaMA 3 70B model for advanced NLP processing
    
    This implementation leverages Groq's powerful model capabilities for:
    - Context-aware summarization
    - Semantic similarity detection
    - Resume-to-job description alignment
    - Style and tone adaptation
    - Simulated TextRank for keyword extraction
    - Intent-based similarity (not just keyword matching)
    """
    api_key = get_groq_api_key()
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Use LLaMA 3 70B (8192 context window) - Groq's most powerful model
    # for advanced semantic understanding and NLP capabilities
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,  # Lower temperature for more consistent, focused results
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

def calculate_semantic_matching_score(resume_text, job_description):
    """
    Use Groq API to calculate a semantic matching score between resume and job description
    using advanced analysis techniques
    """
    system_prompt = """You are an AI trained to evaluate how well a resume matches a job description.
Your goal is to provide a detailed analysis with an overall percentage match score without any explanatory text about your methodology."""
    
    prompt = f"""ANALYZE THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Provide a detailed match analysis that includes:

1. An exact percentage score (e.g., 78%) representing how well the resume matches the job
2. Key strengths of the resume relative to the job description
3. Gap areas where the resume could be improved
4. Analysis of keyword matching and relevance

Important formatting instructions:
- Begin with the match score in this exact format: "MATCH SCORE: XX%"
- Do NOT include any explanation about how you calculated the score
- Do NOT use bullet points, asterisks or other symbols
- Structure the analysis with these exact headings:
  "Key strengths:"
  "Gap areas:"
  "Keyword analysis:"
- Do NOT include any introductory or concluding text
- Do NOT mention your AI capabilities or methodology
- Focus solely on the resume and job description content"""

    try:
        result = call_groq_api(prompt, system_prompt=system_prompt, max_tokens=1000, temperature=0.2)
        return result
    except Exception as e:
        logger.error(f"Error calculating semantic matching score: {str(e)}")
        return "Unable to calculate matching score at this time. Please try again later."

def get_improvement_suggestions(resume_text, job_description):
    """
    Use Groq API to get improvement suggestions with advanced NLP techniques
    based on modern ATS compliance standards and semantic alignment
    """
    system_prompt = """You are an expert resume optimization assistant with advanced NLP capabilities.
Your job is to provide specific, actionable suggestions to improve a resume's match score, without any introductory text or explanations about your methodology."""

    prompt = f"""ANALYZE THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Provide 4-6 specific, actionable suggestions to improve this resume's ATS match score. For each suggestion:
1. Identify the specific issue in the resume
2. Explain why it's a problem for ATS matching
3. Provide a concrete example of how to fix it

Important formatting instructions:
- Do NOT use bullet points, symbols, or asterisks in your response
- Do NOT include any explanation about how you generated these suggestions
- Number each suggestion (1., 2., 3., etc.)
- Include a clear heading for each suggestion
- For each "before" and "after" example, clearly label them as "BEFORE:" and "AFTER:"
- Do NOT include any introductory or concluding text

For example, format your suggestions like this:

1. SUGGESTION HEADING
The specific issue in detail. 
Why this matters for ATS matching.
BEFORE: [original text]
AFTER: [improved text]

2. SUGGESTION HEADING
[and so on...]"""

    try:
        suggestions = call_groq_api(prompt, system_prompt=system_prompt, max_tokens=1200, temperature=0.3)
        return suggestions
    except Exception as e:
        logger.error(f"Error getting improvement suggestions: {str(e)}")
        return "Unable to generate suggestions at this time. Please try again later."

def rewrite_resume(resume_text, job_description):
    """
    Use Groq API to rewrite the resume with advanced NLP techniques
    while preserving the original layout and structure
    """
    system_prompt = """You are an expert resume optimization AI focused on making minimal but effective changes.
Your task is to preserve the original resume's layout while making targeted text changes to improve ATS match score."""
    
    prompt = f"""CAREFULLY ENHANCE THIS RESUME TO BETTER MATCH THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME:
{resume_text}

Important instructions:
1. Preserve the EXACT layout/structure of the original resume including all formatting
2. Maintain ALL section headings exactly as they appear in the original
3. Make targeted text improvements without changing the resume's structure
4. Keep ALL contact information and personal details unchanged
5. DO NOT add any introduction or conclusion text (like "Here is the rewritten resume...")
6. DO NOT include ANY symbols (*, +, •, etc.) or formatting markers
7. DO NOT add any explanation text about how the resume was optimized

Make ONLY these types of subtle text improvements:
- Replace generic terms with specific keywords from the job description
- Enhance action verbs to be more impactful and relevant
- Improve phrasing to better match the intent of the job description
- Highlight relevant skills and experiences that match the job requirements
- Ensure proper tense consistency throughout

Do NOT:
- Change the overall structure or layout
- Add new sections or bullet points
- Remove existing content
- Include any explanation about the optimization process
- Add any decorative elements or special characters
- Include any introductory or closing text

Provide JUST the enhanced resume text that maintains the exact layout of the original with subtle but effective text improvements."""

    try:
        rewritten_resume = call_groq_api(prompt, system_prompt=system_prompt, max_tokens=2000, temperature=0.2)
        return rewritten_resume
    except Exception as e:
        logger.error(f"Error rewriting resume: {str(e)}")
        return "Unable to rewrite resume at this time. Please try again later."
