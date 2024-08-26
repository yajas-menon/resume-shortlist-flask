# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
# from transformers import pipeline
# import pdfplumber
# import re
# import spacy

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# # Load pre-trained BERT model pipeline for text classification
# nlp_bert = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# # Load spaCy English large model for NER
# nlp_spacy = spacy.load("en_core_web_lg")

# # Regular expression patterns for email and phone number
# EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
# PHONE_PATTERN = re.compile(r'(\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text()
#     return text

# def extract_personal_info(resume_text):
#     # Use spaCy to extract named entities for full name
#     doc = nlp_spacy(resume_text)
#     full_name = ""
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             full_name = ent.text
#             break  # Assuming the first PERSON entity is the candidate's full name

#     # Extract email using regex
#     email_match = EMAIL_PATTERN.search(resume_text)
#     email = email_match.group(0) if email_match else "Not found"

#     # Extract phone number using regex
#     phone_match = PHONE_PATTERN.search(resume_text)
#     phone = phone_match.group(0) if phone_match else "Not found"

#     return full_name, email, phone

# def extract_skills_and_intent(resume_text):
#     # Use spaCy NER to dynamically identify skills
#     skills = set()
#     doc = nlp_spacy(resume_text)
#     for chunk in doc.noun_chunks:
#         skills.add(chunk.text.lower())

#     # Extract the objectives section using regex
#     objectives_match = re.search(r'(objective|career objective|professional summary)(.*?)(experience|education|skills)', 
#                                  resume_text, re.IGNORECASE | re.DOTALL)
#     objectives = objectives_match.group(2).strip() if objectives_match else "Not found"

#     return {
#         "skills": list(skills),
#         "objectives": objectives
#     }

# @app.route('/upload', methods=['POST'])
# @cross_origin()
# def upload_resume():
#     job_description = request.form['job_description']
#     file = request.files['file']

#     # Extract text from PDF
#     resume_text = extract_text_from_pdf(file)

#     # Extract personal information
#     full_name, email, phone = extract_personal_info(resume_text)

#     # Extract skills and intent from resume
#     extracted_data = extract_skills_and_intent(resume_text)

#     # Use BERT to analyze resume text against job description
#     result = nlp_bert(resume_text, [job_description])

#     # Calculate fit score
#     fit_score = result['scores'][0] * 100

#     # Response with the result
#     response = {
#         "full_name": full_name,
#         "email": email,
#         "phone": phone,
#         "skills": extracted_data["skills"],
#         "objectives": extracted_data["objectives"],
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
import spacy

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load pre-trained BERT model pipeline for text classification
nlp_bert = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nlp = spacy.load("en_core_web_lg")

# Load larger spaCy English model for better NER performance
nlp_spacy = spacy.load("en_core_web_lg")

# Regular expression patterns for email and phone number
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'(\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_personal_info(resume_text):
    # Use spaCy to extract named entities for first and last name
    doc = nlp_spacy(resume_text)
    full_name = ""
    first_name = ""
    last_name = ""
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            full_name = ent.text
            # Split the full name into first and last name
            name_parts = full_name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = name_parts[-1]
            elif len(name_parts) == 1:
                first_name = name_parts[0]
            break  # Assuming the first PERSON entity is the candidate's name

    # Extract email using regex
    email_match = EMAIL_PATTERN.search(resume_text)
    email = email_match.group(0) if email_match else "Not found"

    # Extract phone number using regex
    phone_match = PHONE_PATTERN.search(resume_text)
    phone = phone_match.group(0) if phone_match else "Not found"

    return full_name, first_name, last_name, email, phone

def extract_skills_and_intent(resume_text):
    # Initialize SpaCy on the resume text
    doc = nlp(resume_text)

    # Create a set to hold unique skills
    skills = set()

    # Iterate over recognized entities
    for ent in doc.ents:
        # Check if the entity label is relevant to skills or technologies
        if ent.label_ in ["ORG", "PRODUCT", "SKILL", "TECHNOLOGY"]:
            skills.add(ent.text.strip())

    # Convert the set of skills to a list
    skills = list(skills)

    # Extract the objectives section using regex
    objectives_match = re.search(r'(objective|career objective|professional summary)(.*?)(experience|education|skills)', 
                                 resume_text, re.IGNORECASE | re.DOTALL)
    objectives = objectives_match.group(2).strip() if objectives_match else "Not found"

    return {
        "skills": skills,
        "objectives": objectives
    }

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_resume():
    job_description = request.form['job_description']
    file = request.files['file']

    # Extract text from PDF
    resume_text = extract_text_from_pdf(file)

    # Extract personal information
    full_name, first_name, last_name, email, phone = extract_personal_info(resume_text)

    # Extract skills and intent from resume
    extracted_data = extract_skills_and_intent(resume_text)

    # Use BERT to analyze resume text against job description
    result = nlp_bert(resume_text, [job_description])

    # Calculate fit score
    fit_score = result['scores'][0] * 100

    # Response with the result
    response = {
        "full_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "skills": extracted_data["skills"],
        "objectives": extracted_data["objectives"],
        "fit_score": fit_score,
        "label": result['labels'][0],
        "description": job_description,
        "resume_text": resume_text
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
