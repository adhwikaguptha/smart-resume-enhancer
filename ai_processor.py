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
    using techniques that simulate advanced transformer-based scoring systems
    """
    system_prompt = """You are an AI trained to evaluate how well a resume matches a job description by simulating these advanced NLP techniques:
1. TextRank-like summarization to identify central phrases and concepts
2. Semantic similarity detection between different phrasings with similar intent
3. Transformer-based scoring simulations like those used in modern ATS systems
4. Contextual keyword relevance assessment beyond simple word-matching

Your goal is to provide a detailed analysis with an overall percentage match score."""
    
    prompt = f"""ANALYZE THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Simulate a sophisticated ATS scoring system by analyzing:

1. Keyword Optimization:
   - Identify and extract key terms from the job description
   - Assess how well these appear in the resume (exact matches and semantic equivalents)
   - Consider keyword placement and density (headers, summary, experience sections)

2. Semantic Matching:
   - Detect paraphrased concepts and skills (not just exact keyword matches)
   - Evaluate alignment between job requirements and candidate's experience
   - Consider contextual relevance of skills mentioned

3. ATS Scoring Simulation:
   - Simulate how actual ATS systems would score based on:
     * Keyword density and placement
     * Phrase alignment with job requirements
     * Structure and formatting friendliness 
     * Action verb usage and appropriateness

4. Role-Specific Evaluation:
   - Assess match for both technical skills and soft skills
   - Consider both stated and implied qualifications
   - Evaluate overall career trajectory alignment with role

Format your response as:
MATCH SCORE: XX%

ANALYSIS:
- Key strengths: [list key matching points]
- Gap areas: [list key missing elements]
- Keyword analysis: [analysis of key terms, their placement, and contextual relevance]"""

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
    system_prompt = """You are an expert resume optimization assistant with advanced NLP capabilities:
1. Part-of-speech tagging to verify verb usage (e.g., "developed UI" > "UI Developer")
2. Semantic alignment to match resume content with ATS and recruiter intent patterns
3. Keyword extraction directly from the job description
4. Simulated ATS scoring based on systems like Workday, Greenhouse, Lever, and Taleo
5. Context-aware summarization to identify key elements
6. Resume-to-job description alignment through semantic similarity

Your job is to provide specific, actionable suggestions to improve a resume's match score."""

    prompt = f"""ANALYZE THE FOLLOWING RESUME AGAINST THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Provide 4-6 specific, actionable suggestions to improve this resume's ATS match score. For each suggestion:
1. Identify the specific issue in the resume
2. Explain why it's a problem for ATS matching
3. Provide a concrete example of how to fix it

Apply these advanced ATS optimization techniques:
- Keyword Optimization: Extract and match keywords directly from the job description
- ATS Compliance: Focus on simple formatting, clear section headings, and parsable structure
- Semantic Alignment: Reword content to align with how ATS systems scan for intent
- Modern Resume Style: Use concise action-oriented language focusing on outcomes
- Role Fit & Human Appeal: Align tone with what the human hiring manager wants to see

Format each suggestion with bullet points and provide specific examples. For each suggestion, include:
- The issue
- Why it matters for ATS
- A "Before" and "After" example showing the improvement"""

    try:
        suggestions = call_groq_api(prompt, system_prompt=system_prompt, max_tokens=1200, temperature=0.3)
        return suggestions
    except Exception as e:
        logger.error(f"Error getting improvement suggestions: {str(e)}")
        return "Unable to generate suggestions at this time. Please try again later."

def rewrite_resume(resume_text, job_description):
    """
    Use Groq API to rewrite the resume with advanced NLP techniques
    that simulate capabilities of modern AI systems while using only Groq's LLaMA 3 model
    """
    system_prompt = """You are an expert resume optimization AI, simulating these advanced capabilities:
1. Context-aware summarization - identifying key elements from both resume and job description
2. Semantic similarity detection - understanding relationships between different phrasings
3. Resume-to-JD alignment - transforming content to match job requirements
4. Style and tone adaptation - optimizing language for both ATS and human readers
5. Chain-of-thought reasoning - mapping cause-effect relationships in professional experiences

Your task is to rewrite resumes to maximize their ATS match scores while maintaining authenticity."""
    
    prompt = f"""REWRITE THE FOLLOWING RESUME TO BETTER MATCH THIS JOB DESCRIPTION:

JOB DESCRIPTION:
{job_description}

ORIGINAL RESUME:
{resume_text}

Rewrite this resume to maximize its ATS match score while maintaining the person's honest work history and qualifications. Apply these specific techniques:

1. Keyword Optimization:
   - Extract keywords directly from the job description (e.g., technical skills, soft skills, industry terms)
   - Match them with the candidate's skills and experiences
   - Integrate them naturally throughout the resume

2. ATS Compliance:
   - Use simple formatting with clear section headings (e.g., EXPERIENCE, SKILLS, EDUCATION)
   - Create bullet points for easy parsing by ATS systems
   - Avoid complex formatting that might confuse parsers

3. Semantic Alignment:
   - Reword content to match how ATS systems scan for intent
   - Example: Change "worked on UI" to "translated design into code" if that's what the job requires
   - Align terminology with industry standards mentioned in the job description

4. Modern Resume Style:
   - Use concise, action-oriented language focused on achievements
   - Begin bullet points with strong action verbs
   - Quantify achievements where possible (numbers, percentages, metrics)

5. Role Fit & Human Appeal:
   - Align tone with what hiring managers seek: curiosity, adaptability, initiative
   - Highlight experiences most relevant to the specific role
   - Balance technical details with demonstration of soft skills

Guidelines:
- Maintain the same overall structure and all factual information
- Preserve all contact information and personal details
- Do NOT invent work experience or qualifications not mentioned in the original
- Format as a complete document ready for submission

The goal is to simulate what advanced AI systems can do using only LLaMA 3's native capabilities for semantic understanding."""

    try:
        rewritten_resume = call_groq_api(prompt, system_prompt=system_prompt, max_tokens=2000, temperature=0.2)
        return rewritten_resume
    except Exception as e:
        logger.error(f"Error rewriting resume: {str(e)}")
        return "Unable to rewrite resume at this time. Please try again later."
