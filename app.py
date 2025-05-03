import os
import logging
from flask import Flask, render_template, request, jsonify, flash, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
import tempfile
import uuid
import document_processor
import ai_processor
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
                
                # Calculate initial ATS score
                initial_score = document_processor.calculate_ats_score(resume_text, job_description)
                initial_score_normalized = int(initial_score * 100)
                
                # Get AI suggestions
                suggestions = ai_processor.get_improvement_suggestions(resume_text, job_description)
                
                # Get rewritten resume
                rewritten_resume = ai_processor.rewrite_resume(resume_text, job_description)
                
                # Calculate new ATS score
                new_score = document_processor.calculate_ats_score(rewritten_resume, job_description)
                new_score_normalized = int(new_score * 100)
                
                # Store data in session
                session['resume_text'] = resume_text
                session['rewritten_resume'] = rewritten_resume
                session['job_description'] = job_description
                session['initial_score'] = initial_score_normalized
                session['new_score'] = new_score_normalized
                session['suggestions'] = suggestions
                
                return render_template('index.html', 
                                      initial_score=initial_score_normalized,
                                      new_score=new_score_normalized,
                                      suggestions=suggestions,
                                      rewritten_resume=rewritten_resume,
                                      resume_text=resume_text,
                                      job_description=job_description,
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

@app.route('/download/<format>')
def download_resume(format):
    if 'rewritten_resume' not in session:
        flash('No resume data available. Please analyze a resume first.')
        return redirect(url_for('index'))
    
    rewritten_resume = session.get('rewritten_resume')
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
