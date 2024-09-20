import os
import gradio as gr
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from groq import Groq
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account
import resend
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Get the API keys
groq_api_key = os.getenv("GROQ_API_KEY")
google_sheets_creds_file = os.getenv("GOOGLE_SHEETS_CREDS_FILE")
google_sheet_id = os.getenv("GOOGLE_SHEET_ID")
resend_api_key = os.getenv("RESEND_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

if not google_sheets_creds_file or not google_sheet_id:
    raise ValueError("Google Sheets credentials or Sheet ID are not set in the .env file")

if not resend_api_key:
    raise ValueError("RESEND_API_KEY is not set in the .env file")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

client = Groq(api_key=groq_api_key)
resend.api_key = resend_api_key

print("Groq client initialized")

# Email Template Engine
class EmailTemplate:
    def __init__(self, name, template, required_fields):
        self.name = name
        self.template = template
        self.required_fields = required_fields

class EmailTemplateEngine:
    def __init__(self):
        self.templates = {
            "job_application": EmailTemplate(
                "Job Application",
                """
                Dear {hiring_manager_name},

                I am writing to express my strong interest in the {job_title} position at {company_name}. With my background in {relevant_skills}, I believe I would be a valuable asset to your team.

                {experience_summary}

                Some of my key achievements include:
                {key_achievements}

                I am particularly drawn to {company_name} because {reason_for_interest}. I am excited about the opportunity to bring my skills and passion to your team.

                Thank you for considering my application. I look forward to the opportunity to discuss how I can contribute to {company_name}'s success.

                Best regards,
                {applicant_name}
                """,
                ["hiring_manager_name", "job_title", "company_name", "relevant_skills", "experience_summary", "key_achievements", "reason_for_interest", "applicant_name"]
            ),
            "sales_pitch": EmailTemplate(
                "Sales Pitch",
                """
                Dear {recipient_name},

                I hope this email finds you well. I'm reaching out because {reason_for_contact}. Our product, {product_name}, can help you {main_benefit}.

                Some key features of {product_name} include:
                {key_features}

                Our customers have seen remarkable results, such as:
                {customer_results}

                I'd love to schedule a call to discuss how {product_name} can specifically benefit {company_name}. Are you available for a quick 15-minute chat this week?

                Looking forward to the opportunity to help {company_name} achieve its goals.

                Best regards,
                {sender_name}
                {sender_position}
                {sender_company}
                """,
                ["recipient_name", "reason_for_contact", "product_name", "main_benefit", "key_features", "customer_results", "company_name", "sender_name", "sender_position", "sender_company"]
            )
        }
        self.client = Groq(api_key=groq_api_key)

    def get_template_names(self):
        return list(self.templates.keys())

    def get_template(self, template_name):
        return self.templates.get(template_name)

    def get_required_fields(self, template_name):
        template = self.get_template(template_name)
        return template.required_fields if template else []

    def customize_template(self, template_name, **kwargs):
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        prompt = template.template.format(**kwargs)
        
        response = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


# Set up Google Sheets API
creds = service_account.Credentials.from_service_account_file(
    google_sheets_creds_file, scopes=['https://www.googleapis.com/auth/spreadsheets']
)
sheets_service = build('sheets', 'v4', credentials=creds)

def save_to_google_sheets(data):
    sheet = sheets_service.spreadsheets()
    result = sheet.values().append(
        spreadsheetId=google_sheet_id,
        range='Sheet1',
        valueInputOption='USER_ENTERED',
        insertDataOption='INSERT_ROWS',
        body={'values': [data]}
    ).execute()
    print(f"{result.get('updates').get('updatedCells')} cells appended.")

def create_html_email(content, sender_name, sender_email):
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Professional Email</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .footer {{
                border-top: 1px solid #ddd;
                padding-top: 10px;
                margin-top: 20px;
                font-size: 0.9em;
                color: #666;
            }}
            .content {{
                white-space: pre-wrap;
                font-family: Arial, sans-serif;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Professional Communication</h2>
        </div>
        <div class="content">
{content}
        </div>
        <div class="footer">
            <p>Best regards,<br>{sender_name}<br>{sender_email}</p>
        </div>
    </body>
    </html>
    """
    return html_template

def send_email(from_name, from_email, to_email, subject, content):
    try:
        html_content = create_html_email(content, from_name, from_email)
        response = resend.Emails.send({
            "from": f"{from_name} <onboarding@resend.dev>",
            "to": to_email,
            "subject": subject,
            "html": html_content,
            "reply_to": from_email
        })
        print(f"Email sent. ID: {response['id']}")
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract all text from paragraphs
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        
        # If there's not enough content in paragraphs, get all text
        if len(text) < 1000:
            text = soup.get_text()
        
        return text[:5000]  # Limit to 5000 characters
    except Exception as e:
        return f"Error scraping website: {str(e)}"

def generate_cold_email(template_name, **kwargs):
    if template_name == "job_application":
        # Use the existing logic for job applications
        skills = kwargs.get('relevant_skills', '')
        job_url = kwargs.get('job_url', '')
        website_context = scrape_website(job_url)
        
        # Analyze the job, generate growth ideas, and suggest content improvements
        analysis_prompt = f"""
        Based on the following job description and website content:
        {website_context}

        1. Extract 3-5 key required skills for this position.
        2. Identify 2-3 of the applicant's skills that best match the required skills.
        3. Provide 3 specific, detailed examples of how the applicant's skills can be used to help the organization grow or improve.
        4. Suggest 2 improvements for character development or storylines, using specific examples from the website content. These suggestions should demonstrate how the applicant's skills can be applied to make the characters or narratives more fascinating.

        Applicant's Skills: {skills}

        Format the response as:
        Required Skills: skill1, skill2, skill3, ...

        Matching Skills: skill1, skill2, skill3

        Growth Ideas:
        1. [Detailed example 1]
        2. [Detailed example 2]
        3. [Detailed example 3]

        Character/Storyline Improvement Suggestions:
        1. [Detailed suggestion 1, incorporating applicant's skills and website content]
        2. [Detailed suggestion 2, incorporating applicant's skills and website content]
        """

        analysis_response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "user", "content": analysis_prompt}
            ]
        )
        analysis_result = analysis_response.choices[0].message.content

        # Update kwargs with analysis results
        kwargs.update({
            'experience_summary': analysis_result,
            'key_achievements': "- " + "\n- ".join(analysis_result.split("\n\n")[2].split("\n")[1:]),
            'reason_for_interest': analysis_result.split("\n\n")[-1]
        })

    return template_engine.customize_template(template_name, **kwargs)

template_engine = EmailTemplateEngine()

def chat_function(message, history):
    if not history:
        return "Please use the 'Generate Email' button to create your initial email."
    else:
        # This is an edit request
        full_context = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in history])
        latest_email = history[0][1].split("\n\n")[-2]  # Get the latest version of the email
        
        edit_prompt = f"""
        Original Email (or Latest Version):
        {latest_email}

        Edit History:
        {full_context}

        User's Latest Edit Request:
        {message}

        Please provide an updated version of the email incorporating the user's edit request. 
        Maintain the overall structure and key points of the original email while making the requested changes.
        Consider all previous edits and requests when making this update.
        """

        edit_response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are an AI assistant helping to refine an email."},
                {"role": "user", "content": edit_prompt}
            ]
        )
        edited_email = edit_response.choices[0].message.content
        return f"Here's the updated version of your email based on your request:\n\n{edited_email}\n\nYou can continue to edit this email by sending more edit requests."

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Advanced Personalized Email Generator (Groq - Mixtral)")
    
    with gr.Row():
        name_input = gr.Textbox(label="Your Name")
        email_input = gr.Textbox(label="Your Email")
    
    template_select = gr.Dropdown(choices=template_engine.get_template_names(), label="Select Email Template")
    
    with gr.Row():
        skills_input = gr.Textbox(label="Your Skills (comma-separated)")
        url_input = gr.Textbox(label="Company/Job URL")
    
    recipient_email = gr.Textbox(label="Recipient's Email")
    
    dynamic_inputs = gr.JSON(label="Additional Fields (will update based on selected template)")
    
    generate_button = gr.Button("Generate Email")
    send_button = gr.Button("Send Email")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Edit Request")
    clear = gr.Button("Clear Chat")

    gr.Markdown("""
    1. Select an email template (Job Application or Sales Pitch).
    2. Fill in the required fields, including the dynamic fields that appear based on your template selection.
    3. Click 'Generate Email' to create your initial email and save your data.
    4. Use the chat interface below to refine and edit your email as many times as you like.
    5. Enter the recipient's email address.
    6. When you're satisfied with the email, click 'Send Email' to send it via Resend.
    """)

    def update_dynamic_inputs(template_name):
        required_fields = template_engine.get_required_fields(template_name)
        return {field: "" for field in required_fields if field not in ["sender_name", "sender_email", "recipient_email"]}

    def generate_email_action(template_name, name, email, skills, url, dynamic_inputs):
        try:
            fields = {
                "sender_name": name,
                "sender_email": email,
                "relevant_skills": skills,
                "job_url": url,
                **dynamic_inputs
            }
            generated_email = generate_cold_email(template_name, **fields)
            save_to_google_sheets([name, email, template_name, skills, url])
            return [[None, f"Here's your generated email:\n\n{generated_email}\n\nYou can now edit this email by sending your edit requests."]]
        except Exception as e:
            return [[None, f"Error generating email: {str(e)}"]]

    def send_email_action(name, email, recipient, chatbot):
        if not chatbot:
            return "Please generate an email first."
        if not recipient:
            return "Please enter the recipient's email address."
        
        # Extract the latest email content
        latest_email_message = chatbot[-1][1]
        email_content_start = latest_email_message.find("Here's your generated email:") + len("Here's your generated email:")
        email_content_end = latest_email_message.find("\n\nYou can now edit this email")
        if email_content_end == -1:  # If not found, take the rest of the message
            email_content_end = len(latest_email_message)
        
        email_content = latest_email_message[email_content_start:email_content_end].strip()
        
        subject = "Professional Communication"
        success = send_email(name, email, recipient, subject, email_content)
        if success:
            return "Email sent successfully!"
        else:
            return "Failed to send email. Please try again."

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = chat_function(user_message, history[:-1])
        history[-1][1] = bot_message
        return history

    template_select.change(update_dynamic_inputs, inputs=[template_select], outputs=[dynamic_inputs])
    
    generate_button.click(
        generate_email_action,
        inputs=[template_select, name_input, email_input, skills_input, url_input, dynamic_inputs],
        outputs=[chatbot]
    )
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    send_button.click(send_email_action, inputs=[name_input, email_input, recipient_email, chatbot], outputs=gr.Textbox())
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)
