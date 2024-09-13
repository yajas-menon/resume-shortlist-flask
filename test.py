from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import pipeline
import pdfplumber
import re
import spacy
import json
from groq import Groq

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load pre-trained BERT model pipeline for text classification
nlp_bert = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load larger spaCy English model for better NER performance
nlp_spacy = spacy.load("en_core_web_lg")

# Regular expression patterns for email and phone number
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'(\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

client = Groq(api_key="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")
# Set up OpenAI API key
# openai.api_key = 'sk-6e4BzDsiLC78tNavsBV1VJymxPA5ogWTUhiguLyxtMT3BlbkFJFiHlQwCw_2qxnp211SMn3XsjUKt8Xvs2maXHh71BMA'

def extract_text_from_pdf(file):
    """Extract text from PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text.strip()

def extract_personal_info(resume_text):
    """Extract personal details like name, email, phone."""
    doc = nlp_spacy(resume_text)
    full_name = ""
    first_name = ""
    last_name = ""

    # Extract name from PERSON entity (first occurrence)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            full_name = ent.text
            name_parts = full_name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = name_parts[-1]
            break

    # Extract email and phone using regex
    email_match = EMAIL_PATTERN.search(resume_text)
    email = email_match.group(0) if email_match else "Not found"
    
    phone_match = PHONE_PATTERN.search(resume_text)
    phone = phone_match.group(0) if phone_match else "Not found"

    return full_name, first_name, last_name, email, phone

def extract_skills_and_intent(resume_text):
    """Extract skills and objectives/intents from the resume."""
    doc = nlp_spacy(resume_text)

    # Extracting skills by scanning for key technical and organizational terms
    skills = set()
    skill_keywords = ["Python", "React", "Node.js", "Machine Learning", "SQL", "JavaScript", "Flask", "Docker", "Azure", "API"]
    
    for token in doc:
        if token.text in skill_keywords:
            skills.add(token.text)
    
    # Extracting objectives based on known section titles
    objectives_match = re.search(r'(objective|career objective|professional summary)(.*?)(experience|education|skills)', 
                                 resume_text, re.IGNORECASE | re.DOTALL)
    objectives = objectives_match.group(2).strip() if objectives_match else "Not found"
    
    return {
        "skills": list(skills),
        "objectives": objectives
    }

def extract_education_details_with_gpt(resume_text):
    prompt = f"Extract the education details from the following resume text: {resume_text}. The output should include degree, institution, and year in JSON format as key value pairs only."
    
    # Directly call the OpenAI API here
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # You can switch to "gpt-4" if needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.5,
    )
    resume_text = response.choices[0].message.content.strip()

    try:
        return json.loads(resume_text)
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from GPT response"}

def extract_experience_details_with_gpt(resume_text):
    prompt = f"Extract the work experience details from the following resume text: {resume_text}. The output should include job_title, company, start_year, and end_year in JSON format as key value pairs only."
    
    # Directly call the OpenAI API here
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # You can switch to "gpt-4" if needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.5,
    )
    resume_text = response.choices[0].message.content.strip()

    try:
        return json.loads(resume_text)
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from GPT response"}

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_resume():
    """Handle resume upload and extract details."""
    job_description = request.form['job_description']
    file = request.files['file']

    # Extract text from PDF
    resume_text = extract_text_from_pdf(file)

    # Extract personal info
    full_name, first_name, last_name, email, phone = extract_personal_info(resume_text)

    # Extract skills and intent
    extracted_data = extract_skills_and_intent(resume_text)

    # Extract education and experience details using GPT
    education_details = extract_education_details_with_gpt(resume_text)
    experience_details = extract_experience_details_with_gpt(resume_text)

    # Use BERT to analyze resume text against job description
    result = nlp_bert(resume_text, [job_description])

    # Calculate fit score
    fit_score = result['scores'][0] * 100

    # Construct response
    response = {
        "full_name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "skills": extracted_data["skills"],
        "objectives": extracted_data["objectives"],
        "education": education_details,
        "experience": experience_details,
        "fit_score": fit_score,
        "label": result['labels'][0],
        "description": job_description,
        "resume_text": resume_text
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
