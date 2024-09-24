# import os
# import time
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.environ["API_KEY"])

# def upload_to_gemini(path, mime_type=None):
#   """Uploads the given file to Gemini.

#   See https://ai.google.dev/gemini-api/docs/prompting_with_media
#   """
#   file = genai.upload_file(path, mime_type=mime_type)
#   print(f"Uploaded file '{file.display_name}' as: {file.uri}")
#   return file

# def wait_for_files_active(files):
#   """Waits for the given files to be active.

#   Some files uploaded to the Gemini API need to be processed before they can be
#   used as prompt inputs. The status can be seen by querying the file's "state"
#   field.

#   This implementation uses a simple blocking polling loop. Production code
#   should probably employ a more sophisticated approach.
#   """
#   print("Waiting for file processing...")
#   for name in (file.name for file in files):
#     file = genai.get_file(name)
#     while file.state.name == "PROCESSING":
#       print(".", end="", flush=True)
#       time.sleep(10)
#       file = genai.get_file(name)
#     if file.state.name != "ACTIVE":
#       raise Exception(f"File {file.name} failed to process")
#   print("...all files ready")
#   print()

# # Create the model
# generation_config = {
#   "temperature": 1,
#   "top_p": 0.95,
#   "top_k": 64,
#   "max_output_tokens": 8192,
#   "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#   model_name="gemini-1.5-flash",
#   generation_config=generation_config,
#   # safety_settings = Adjust safety settings
#   # See https://ai.google.dev/gemini-api/docs/safety-settings
# )

# # TODO Make these files available on the local file system
# # You may need to update the file paths
# files = [
#   upload_to_gemini("Resume (1).pdf", mime_type="application/pdf"),
# ]

# # Some files have a processing delay. Wait for them to be ready.
# wait_for_files_active(files)

# chat_session = model.start_chat(
#   history=[
#     {
#       "role": "user",
#       "parts": [
#         files[0],
#         "from the given resume extract First name , last name , full name , phone number , email , skills , educational details and experience from the resume/cvs\n in key value pairs",
#       ],
#     },
#     {
#       "role": "model",
#       "parts": [
#         "**First Name:** Indu\n**Last Name:** Parmar\n**Full Name:** Indu Parmar\n**Phone Number:** +91-9650797257\n**Email:** indu.parmar@gmail.com\n**Skills:**\n  * HR Business Partner\n  * HR Process Transformation & Operations\n  * Talent Development\n  * Workforce Planning\n  * Compensation Administration\n  * Performance Management\n  * Employee Engagement\n  * Feedback Survey\n  * Career Planning\n  * Talent Reviews\n  * Workforce Analytics\n  * Reporting Process\n  * Automation\n  * Workflow Management\n  * Query Resolution\n  * Compensation Finalization\n  * Payroll Processing\n  * Team Management\n  * Feedback\n  * EPIC Assessment\n  *  Passion Drivers\n  * Data Analysis\n  * Action Planning\n  *  Training & Development\n  * Strategic Initiatives\n  * Communication\n  * Data Reporting\n  * Business Analysis\n  * Project Management\n  *  Presentation Skills\n  *  Management Consulting\n  *  Business Analysis\n  *  Computer Application\n  *  MS Office \n\n\n**Educational Details:**\n*  Certification in Management Consulting, CDC, Govt. of India  (2011)\n*  AHBAP certification for business analysis, HCL (2011)\n*  Master in Computer Application, GGSIP University, New Delhi (2005-2008)\n*  Bachelor in Computer Application, GGSIP University, New Delhi (2002-2005)\n\n**Experience:**\n\n**Sr. Specialist – HR Process Transformation & Operations**\nResponsible for providing operational support to the Centers of Excellence in design & governance of key HR Processes.\nKey Responsibilities:\n* Workflow Management of Employee Life Cycle Applications\n*  Compensation Administration\n* Automation solutions for workforce administration\n* Team management\n\n**Accomplishments:**\n* Implemented automations that saved 125 days of effort and ~81K USD in costs.\n* Achieved a 95% query resolution rate within SLA, with a 2% escalation rate and a 98% first-time resolution rate.\n*  Conducted payroll audits of 37K inputs worth INR 34 crores for Offshore India and 14K inputs worth 17 million USD for Onsite.\n\n\n**Specialist - HR Feedback Survey**\nResponsible for co-creating, managing and driving the assessment for identifying employee passion drivers with Employee Passion Indicative Count (EPIC) assessment\n\nKey Areas of Responsibilities:\n*  Co-designed and conducted EPIC assessment survey.\n*  Created awareness and increased participation of employees in the survey.\n*  Collated data, analyzed it with management dashboard and recommended an action plan through post interventions.\n\n**Accomplishments:**\n*  Participation of 51K employees in the assessment survey and 3K+ through floor walks.\n*  Managed EPIC post interventions, i.e. facilitating EPIC Workshops on \"Creating a Passionate Team.\"\n*  Successfully conceptualized, designed, and executed \"Most Passionate HCL Manager on MEME.\"\n\n**HR Business Partner**\nResponsible for leading and driving implementation of the HR processes & policies for the assigned business line and contribute to achievement of delivery goals in compliance with the policies of HCL Technologies.\nKey Areas of Responsibilities:\n* Workforce planning\n*  Compensation planning & administration\n*  Performance management\n*  Employee engagement\n*  Strategic Initiatives\n*  Workforce Analytics\n\n**Accomplishments:**\n*   End-to-end performance management and compensation administration for 2,600 employees.\n*   Managed workforce analytics for a business unit of 5,800 employees.\n*   Conceptualized and implemented strategic initiatives for 5,800 employees.\n*  Handled a sub-business unit of 600+ with an 11.5% attrition rate, achieving 25% attrition reversals.\n\n**Awards:**\n*  O2 Award'15, HCL Technologies (2015)\n*  Center Stage Award in Live Wire category, HCL Technologies (2015) \n",
#       ],
#     },
#   ]
# )

# response = chat_session.send_message("from the given resume extract First name , last name , full name , phone number , email , skills , educational details and experience from the resume/cvs\n in key value pairs")

# print(response.text)


import os
import time
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin


# Load environment variables
load_dotenv()

# Configure Google Gemini API key
genai.configure(api_key=os.environ["API_KEY"])

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
    print()

def save_to_excel(data, filename):
    """Save JSON data to Excel file."""
    df = pd.DataFrame([data])
    output_path = os.path.join(UPLOAD_FOLDER, filename)
    df.to_excel(output_path, index=False)
    print(f"Data saved to {output_path}")
    return output_path

@app.route('/upload_resume', methods=['POST'])
@cross_origin()
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file to the upload folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Upload to Gemini API
        uploaded_file = upload_to_gemini(file_path, mime_type="application/pdf")
        
        # Wait for file to be processed
        wait_for_files_active([uploaded_file])
        
        # Start chat session with the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        uploaded_file,
                        "from the given resume extract First name (firstName), last name (lastName), full name (fullName), phone number (phoneNumber), email (email), skills (skills), educational details (education) with college or university tier and experience from the resume/cvs strictly in the format of json format only no commas and special characters should be added \n",
                    ],
                }
            ]
        )

        # Send the prompt to extract the data from the resume
        response = chat_session.send_message("from the given resume extract First name (firstName), last name (lastName), full name (fullName), phone number (phoneNumber), email (email), skills (skills), educational details (education) it should give with college or university tier and experience from the resume/cvs strictly in the format of json format only no commas and special characters should be added \n")
        
        # Parse the response as JSON (assuming it's valid JSON)
        extracted_data = response.text
        
        # Convert JSON data to a Python dictionary
        json_data = eval(extracted_data)
        
        # Save the extracted data to Excel
        excel_filename = f"{os.path.splitext(filename)[0]}_extracted_data.xlsx"
        excel_file_path = save_to_excel(json_data, excel_filename)

        # Return the response from the model along with the file path to the Excel file
        print(json_data)
        return jsonify({"extracted_data": json_data, "excel_file": excel_file_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
