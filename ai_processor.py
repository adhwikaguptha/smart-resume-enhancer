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
    Using Groq's advanced LLM capabilities
    """
    api_key = get_groq_api_key()
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Use the most advanced model available on Groq for deep text comprehension
    # We'll use llama3-70b-8192 which provides the best semantic understanding
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,  # Lower temperature for more consistent results
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
    """
    system_prompt = """You are an AI trained to evaluate how well a resume matches a job description using advanced NLP techniques including:
1. Part-of-speech tagging to verify proper use of action verbs
2. Semantic similarity matching for paraphrased terms
3. TF-IDF weighted keyword matching
4. ATS simulation based on industry-standard systems like Workday, Greenhouse, Lever, and Taleo

Provide a detailed analysis with an overall percentage match score."""
    
    prompt = f"""ANALYZE THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Calculate a match score as a percentage and explain your reasoning. Consider the following factors:
1. Required skills and how well they match
2. Experience level and relevance
3. Education requirements
4. Industry-specific terminology
5. Action verbs and their appropriateness
6. Keyword density and placement

Format your response as:
MATCH SCORE: XX%

ANALYSIS:
- Key strengths: [list key matching points]
- Gap areas: [list key missing elements]
- Keyword analysis: [analysis of key terms]"""

    try:
        result = call_groq_api(prompt, system_prompt=system_prompt, max_tokens=1000, temperature=0.2)
        return result
    except Exception as e:
        logger.error(f"Error calculating semantic matching score: {str(e)}")
        return "Unable to calculate matching score at this time. Please try again later."

def get_improvement_suggestions(resume_text, job_description):
    """
    Use Groq API to get improvement suggestions with advanced NLP techniques
    """
    system_prompt = """You are an expert resume analyzer with the following capabilities:
1. Part-of-speech tagging to verify verbs vs. nouns (e.g., "developed UI" > "UI Developer")
2. Semantic similarity matching for paraphrased matches
3. TF-IDF Matching to identify important keywords
4. Simulated ATS engine logic based on systems like Workday, Greenhouse, Lever, and Taleo

Your job is to provide specific, actionable suggestions to improve a resume's match score for a job description."""

    prompt = f"""ANALYZE THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Provide 4-6 specific, actionable suggestions to improve this resume's ATS match score. For each suggestion:
1. Identify the specific issue in the resume
2. Explain why it's a problem for ATS matching
3. Provide a concrete example of how to fix it

Consider:
- Keyword alignment and semantic matching
- Proper use of industry terminology
- Action verb optimization
- Skills presentation and formatting
- Experience description relevance
- Quantifiable achievements
- Education and certification placement

Format each suggestion with bullet points and provide specific examples."""

    try:
        suggestions = call_groq_api(prompt, system_prompt=system_prompt, max_tokens=1000, temperature=0.3)
        return suggestions
    except Exception as e:
        logger.error(f"Error getting improvement suggestions: {str(e)}")
        return "Unable to generate suggestions at this time. Please try again later."

def rewrite_resume(resume_text, job_description):
    """
    Use Groq API to rewrite the resume with advanced NLP techniques
    """
    system_prompt = """You are an expert resume optimization AI with the following capabilities:
1. Deep semantic understanding with unified multimodal architecture
2. Extended context window processing for comprehensive analysis
3. Enhanced few-shot learning to understand implicit role expectations
4. Semantic embedding space for similarity detection between different phrasings
5. Chain-of-thought reasoning to map cause-effect relationships

Your task is to rewrite resumes to maximize ATS match scores while maintaining authenticity."""
    
    prompt = f"""REWRITE THE FOLLOWING RESUME TO BETTER MATCH THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME:
{resume_text}

Rewrite this resume to maximize its ATS match score while maintaining the person's honest work history and qualifications. Apply these advanced techniques:

1. Part-of-speech optimization: Use strong action verbs and proper noun forms
2. Semantic similarity matching: Align resume terminology with job description through paraphrasing
3. TF-IDF weighted keyword placement: Position important terms optimally
4. ATS-friendly formatting: Use standard section headings and structures
5. Quantified achievements: Add metrics where implied but not explicitly stated
6. Role-specific terminology: Incorporate industry-specific language from the job description

Guidelines:
- Maintain the same overall structure
- Preserve all contact information and personal details
- Highlight relevant skills and experience that match the job description
- Incorporate keywords naturally, not as keyword stuffing
- Keep a professional tone and voice
- Do NOT invent work experience or qualifications not mentioned in the original

The rewritten resume should appear as a complete, formatted document ready for submission."""

    try:
        rewritten_resume = call_groq_api(prompt, system_prompt=system_prompt, max_tokens=2000, temperature=0.2)
        return rewritten_resume
    except Exception as e:
        logger.error(f"Error rewriting resume: {str(e)}")
        return "Unable to rewrite resume at this time. Please try again later."
