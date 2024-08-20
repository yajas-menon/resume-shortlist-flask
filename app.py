# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
# from transformers import pipeline
# import pdfplumber
# import re

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# # Load pre-trained BERT model pipeline for text classification
# nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text()
#     return text

# def extract_skills_and_intent(resume_text):
#     # Dummy function for extracting skills - you can enhance it by using NLP techniques
#     skills = re.findall(r'\b[A-Za-z]+\b', resume_text)
#     return skills

# @app.route('/upload', methods=['POST'])
# @cross_origin()
# def upload_resume():
#     job_description = request.form['job_description']
#     file = request.files['file']

#     # Extract text from PDF
#     resume_text = extract_text_from_pdf(file)

#     # Extract skills and intent from resume
#     skills = extract_skills_and_intent(resume_text)

#     # Use BERT to analyze resume text against job description
#     result = nlp(resume_text, [job_description])

#     # Calculate fit score
#     fit_score = result['scores'][0] * 100

#     # Response with the result
#     response = {
#         "skills": skills,
#         "fit_score": fit_score,
#         "label": result['labels'][0],
#         "description": job_description,
#         "resume_text": resume_text
#     }
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import pipeline
import pdfplumber
import re

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load pre-trained BERT model pipeline for text classification
nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Predefined lists for primary, secondary, and soft skills (you can expand these)
PRIMARY_SKILLS = ["Python", "Java", "JavaScript", "React", "Node.js", "SQL", "AWS"]
SECONDARY_SKILLS = ["Django", "Flask", "MongoDB", "GraphQL", "Docker", "Kubernetes"]
SOFT_SKILLS = ["communication", "leadership", "teamwork", "problem-solving", "adaptability"]

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_skills_and_intent(resume_text):
    # Extract skills from the resume text
    primary_skills = [skill for skill in PRIMARY_SKILLS if re.search(rf'\b{skill}\b', resume_text, re.IGNORECASE)]
    secondary_skills = [skill for skill in SECONDARY_SKILLS if re.search(rf'\b{skill}\b', resume_text, re.IGNORECASE)]
    soft_skills = [skill for skill in SOFT_SKILLS if re.search(rf'\b{skill}\b', resume_text, re.IGNORECASE)]

    # Extract the objectives section using regex
    objectives_match = re.search(r'(objective|career objective|professional summary)(.*?)(experience|education|skills)', 
                                 resume_text, re.IGNORECASE | re.DOTALL)
    objectives = objectives_match.group(2).strip() if objectives_match else "Not found"

    return {
        "primary_skills": primary_skills,
        "secondary_skills": secondary_skills,
        "soft_skills": soft_skills,
        "objectives": objectives
    }

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_resume():
    job_description = request.form['job_description']
    file = request.files['file']

    # Extract text from PDF
    resume_text = extract_text_from_pdf(file)

    # Extract skills and intent from resume
    extracted_data = extract_skills_and_intent(resume_text)

    # Use BERT to analyze resume text against job description
    result = nlp(resume_text, [job_description])

    # Calculate fit score
    fit_score = result['scores'][0] * 100

    # Response with the result
    response = {
        "primary_skills": extracted_data["primary_skills"],
        "secondary_skills": extracted_data["secondary_skills"],
        "soft_skills": extracted_data["soft_skills"],
        "objectives": extracted_data["objectives"],
        "fit_score": fit_score,
        "label": result['labels'][0],
        "description": job_description,
        "resume_text": resume_text
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
