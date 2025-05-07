import os
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import docx
import numpy as np
from io import BytesIO
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def pdf_extract_text(file_path):
    """Extract text from a PDF file using pdfminer"""
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
            
    text = fake_file_handle.getvalue()
    
    # Close resources
    converter.close()
    fake_file_handle.close()
    
    return text

def extract_text(file_path):
    """
    Extract text from PDF or DOCX file
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return pdf_extract_text(file_path)
    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def calculate_ats_score(resume_text, job_description):
    """
    Calculate semantic similarity between resume and job description
    Since we don't have sentence-transformers available, we'll use a simple
    keyword-based approach as a fallback
    """
    # Normalize inputs
    resume_text = resume_text.lower()
    job_description = job_description.lower()
    
    # Extract words (excluding common stop words)
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'is', 'are', 'was', 'were'}
    job_desc_words = set([word.strip('.,;:()"\'-') for word in job_description.split() if word.strip('.,;:()"\'-').lower() not in stop_words])
    
    # Count matching words
    matches = 0
    for word in job_desc_words:
        if word and len(word) > 2 and word in resume_text:  # Skip empty and very short words
            matches += 1
    
    # Calculate score (0 to 1)
    if len(job_desc_words) > 0:
        score = matches / len(job_desc_words)
    else:
        score = 0
        
    # Apply some normalization to make score distribution more realistic
    score = np.clip(score * 1.5, 0, 1)  # Inflate score slightly, but cap at 1
    return score

def create_docx(text):
    """
    Create a DOCX document from text
    """
    doc = Document()
    for paragraph in text.split('\n'):
        if paragraph.strip():  # Skip empty paragraphs
            doc.add_paragraph(paragraph)
    
    # Save to BytesIO object
    output = BytesIO()
    doc.save(output)
    return output

def create_pdf(text):
    """
    Create a PDF document from text
    """
    output = BytesIO()
    c = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    
    # Configure text rendering
    c.setFont("Helvetica", 11)
    y_position = height - 50  # Start from top with margin
    line_height = 14
    margin = 50
    usable_width = width - 2 * margin
    
    # Split text into paragraphs and wrap lines
    for paragraph in text.split('\n'):
        if not paragraph.strip():  # Skip empty paragraphs
            y_position -= line_height / 2
            continue
            
        # Simple text wrapping
        words = paragraph.split()
        line = ""
        for word in words:
            test_line = line + " " + word if line else word
            line_width = c.stringWidth(test_line, "Helvetica", 11)
            
            if line_width <= usable_width:
                line = test_line
            else:
                # Draw the line and move down
                c.drawString(margin, y_position, line)
                y_position -= line_height
                line = word
                
                # Check if we need a new page
                if y_position < margin:
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    y_position = height - 50
        
        # Draw the last line of the paragraph
        if line:
            c.drawString(margin, y_position, line)
            y_position -= line_height * 1.5  # Extra space between paragraphs
        
        # Check if we need a new page
        if y_position < margin:
            c.showPage()
            c.setFont("Helvetica", 11)
            y_position = height - 50
    
    c.save()
    return output
