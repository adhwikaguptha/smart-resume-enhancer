document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const uploadForm = document.getElementById('upload-form');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            const resumeFile = document.getElementById('resume').files[0];
            const jobDescription = document.getElementById('job_description').value.trim();
            
            let hasError = false;
            
            // Check if resume is uploaded
            if (!resumeFile) {
                alert('Please upload a resume file');
                hasError = true;
            } else {
                // Check file extension
                const fileExt = resumeFile.name.split('.').pop().toLowerCase();
                if (fileExt !== 'pdf' && fileExt !== 'docx') {
                    alert('Please upload a PDF or DOCX file');
                    hasError = true;
                }
            }
            
            // Check if job description is provided
            if (!jobDescription) {
                alert('Please enter a job description');
                hasError = true;
            }
            
            if (hasError) {
                event.preventDefault();
            } else {
                // Add loading state
                const submitBtn = event.submitter;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
                submitBtn.disabled = true;
            }
        });
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
