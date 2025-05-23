import os
import logging
from flask import Flask, render_template, request, jsonify, flash, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
import tempfile
import uuid
import document_processor
import ai_processor
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add file handler for detailed logging
fh = logging.FileHandler('app.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# Configure Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")

# Configure upload settings
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
TEMP_FOLDER = tempfile.gettempdir()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Generate a unique session ID to track this user's files
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Check if the post request has the file part
        if 'resume' not in request.files:
            flash('No resume file uploaded')
            return redirect(url_for('index'))
        
        resume_file = request.files['resume']
        job_description = request.form.get('job_description', '')
        
        # If user did not select a file
        if resume_file.filename == '':
            flash('No resume file selected')
            return redirect(url_for('index'))
            
        # If job description is empty
        if not job_description:
            flash('Job description is required')
            return redirect(url_for('index'))
        
        # Check if file is allowed
        if resume_file and allowed_file(resume_file.filename):
            # Save file temporarily
            filename = secure_filename(resume_file.filename)
            file_path = os.path.join(TEMP_FOLDER, f"{session_id}_{filename}")
            resume_file.save(file_path)
            session['resume_path'] = file_path
            session['original_filename'] = filename
            
            # Extract text from resume
            try:
                resume_text = document_processor.extract_text(file_path)
                
                # Get detailed match analysis using advanced semantic techniques
                try:
                    match_analysis = ai_processor.calculate_semantic_matching_score(resume_text, job_description)
                    if "Unable to calculate matching score" in match_analysis:
                        logger.warning("Failed to calculate detailed matching score, falling back to basic algorithm")
                        # Fall back to basic algorithm if the advanced one fails
                        initial_score = document_processor.calculate_ats_score(resume_text, job_description)
                        initial_score_normalized = int(initial_score * 100)
                    else:
                        # Extract the score from the analysis (format is "MATCH SCORE: XX%")
                        try:
                            score_line = [line for line in match_analysis.split('\n') if 'MATCH SCORE' in line]
                            if score_line:
                                score_text = score_line[0].split(':')[1].strip()
                                initial_score_normalized = int(score_text.replace('%', ''))
                                initial_score = initial_score_normalized / 100
                            else:
                                # Fall back if we can't parse the score
                                initial_score = document_processor.calculate_ats_score(resume_text, job_description)
                                initial_score_normalized = int(initial_score * 100)
                        except Exception as e:
                            logger.error(f"Error parsing semantic match score: {str(e)}")
                            initial_score = document_processor.calculate_ats_score(resume_text, job_description)
                            initial_score_normalized = int(initial_score * 100)
                except Exception as e:
                    logger.error(f"Error with semantic match analysis: {str(e)}")
                    # Fall back to basic algorithm if the advanced one fails
                    initial_score = document_processor.calculate_ats_score(resume_text, job_description)
                    initial_score_normalized = int(initial_score * 100)
                    match_analysis = "Unable to generate detailed match analysis."
                
                # Get advanced AI suggestions using part-of-speech tagging and semantic matching
                try:
                    suggestions = ai_processor.get_improvement_suggestions(resume_text, job_description)
                    # Check if suggestions contain an error message
                    if "Unable to generate suggestions" in suggestions:
                        logger.warning("Failed to generate suggestions")
                        flash("Unable to generate suggestions. Please try again later.")
                        suggestions = "Unable to generate suggestions at this time."
                except Exception as e:
                    logger.error(f"Error getting suggestions: {str(e)}")
                    suggestions = "Unable to generate suggestions at this time."
                
                # Get improved resume using advanced NLP techniques
                try:
                    rewritten_resume = ai_processor.rewrite_resume(resume_text, job_description)
                    # Check if rewritten_resume contains an error message
                    if "Unable to rewrite resume" in rewritten_resume:
                        logger.warning("Failed to rewrite resume")
                        flash("Unable to rewrite the resume. Please try again later.")
                        rewritten_resume = resume_text  # Use original resume as fallback
                        new_score = initial_score  # Use the same score
                        new_score_normalized = initial_score_normalized
                    else:
                        # Try to get a detailed analysis of the rewritten resume
                        try:
                            new_match_analysis = ai_processor.calculate_semantic_matching_score(rewritten_resume, job_description)
                            score_line = [line for line in new_match_analysis.split('\n') if 'MATCH SCORE' in line]
                            if score_line:
                                score_text = score_line[0].split(':')[1].strip()
                                new_score_normalized = int(score_text.replace('%', ''))
                                new_score = new_score_normalized / 100
                            else:
                                # Fall back to basic algorithm
                                new_score = document_processor.calculate_ats_score(rewritten_resume, job_description)
                                new_score_normalized = int(new_score * 100)
                        except Exception as e:
                            logger.error(f"Error with new semantic match analysis: {str(e)}")
                            # Fall back to basic algorithm
                            new_score = document_processor.calculate_ats_score(rewritten_resume, job_description)
                            new_score_normalized = int(new_score * 100)
                except Exception as e:
                    logger.error(f"Error rewriting resume: {str(e)}")
                    flash("Unable to rewrite the resume. Please try again later.")
                    rewritten_resume = resume_text  # Use original resume as fallback
                    new_score = initial_score  # Use the same score
                    new_score_normalized = initial_score_normalized
                
                # Store large data as temporary files instead of in session
                rewritten_resume_path = os.path.join(TEMP_FOLDER, f"{session_id}_rewritten_resume.txt")
                with open(rewritten_resume_path, 'w') as f:
                    f.write(rewritten_resume)
                
                # Store only paths and metadata in session to avoid cookie size limits
                session['rewritten_resume_path'] = rewritten_resume_path
                session['initial_score'] = initial_score_normalized
                session['new_score'] = new_score_normalized
                
                # Pass the match analysis to the template if available
                if 'match_analysis' in locals() and "Unable to generate" not in match_analysis:
                    has_detailed_analysis = True
                else:
                    has_detailed_analysis = False
                    match_analysis = "Detailed analysis not available."
                
                return render_template('index.html', 
                                      initial_score=initial_score_normalized,
                                      new_score=new_score_normalized,
                                      suggestions=suggestions,
                                      rewritten_resume=rewritten_resume,
                                      resume_text=resume_text,
                                      job_description=job_description,
                                      match_analysis=match_analysis,
                                      has_detailed_analysis=has_detailed_analysis,
                                      analysis_complete=True)
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                flash(f'Error processing file: {str(e)}')
                return redirect(url_for('index'))
        else:
            flash('File type not allowed. Please upload a PDF or DOCX file.')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        flash(f'An unexpected error occurred: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download/<format>', methods=['GET'])
def download_resume(format):
    try:
        if 'rewritten_resume_path' not in session:
            flash('No resume data available. Please analyze a resume first.')
            return redirect(url_for('index'))
        
        rewritten_resume_path = session.get('rewritten_resume_path')
        if not rewritten_resume_path or not os.path.exists(rewritten_resume_path):
            flash('Resume content is missing or expired. Please try analyzing your resume again.')
            return redirect(url_for('index'))
            
        # Read the rewritten resume from the temporary file
        with open(rewritten_resume_path, 'r') as f:
            rewritten_resume = f.read()
            
        original_filename = session.get('original_filename', 'resume')
        base_filename = original_filename.rsplit('.', 1)[0]
        
        # Create the file in memory
        if format == 'docx':
            output = document_processor.create_docx(rewritten_resume)
            mimetype = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            filename = f"{base_filename}_rewritten.docx"
        elif format == 'pdf':
            output = document_processor.create_pdf(rewritten_resume)
            mimetype = 'application/pdf'
            filename = f"{base_filename}_rewritten.pdf"
        else:
            flash('Invalid format specified')
            return redirect(url_for('index'))
            
        # Create a file-like object
        file_obj = io.BytesIO(output.getvalue())
        file_obj.seek(0)
        
        return send_file(
            file_obj,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error generating downloadable file: {str(e)}")
        flash('An error occurred while generating the downloadable file. Please try again.')
        return redirect(url_for('index'))

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('The file is too large')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_server_error(error):
    flash('An internal server error occurred')
    return redirect(url_for('index')), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    flash('An unexpected error occurred')
    return redirect(url_for('index')), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
