import json
import re
import os
import base64
import ollama
from email.mime.text import MIMEText
from dash import Dash, html, dcc, callback, Output, Input, State
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import find_dotenv, load_dotenv

# Force GPU usage for Ollama (if applicable)
os.environ["OLLAMA_FORCE_GPU"] = "1"

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Gmail API Setup
try:
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    tools = toolkit.get_tools()
except Exception as e:
    print(f"⚠️ Error initializing Gmail API: {e}")
    api_resource = None  # Ensures the app doesn’t crash

# Model Name (DeepSeek-R1 via Ollama)
MODEL_NAME = "deepseek-r1:1.5b"

def clean_response(response):
    """Extracts only the email content from the response, removing <think> reasoning."""
    match = re.search(r"(Subject:.*)", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()

def process_chat(user_input, chat_history):
    """Generates an AI-crafted email response."""
    if not user_input:
        return "⚠️ Error: No user input provided."

    messages = [
        {"role": "system", "content": (
            "You are an AI email assistant. Generate clear, professional, and concise emails with a formal tone. "
            "ONLY include the email subject, greeting, body, and closing. "
            "DO NOT assume any details that the user has not provided."
        )}
    ]

    # Append chat history
    for msg in chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages.append({"role": role, "content": msg.content})

    messages.append({"role": "user", "content": user_input})

    # Get response from Ollama
    response = ollama.chat(model=MODEL_NAME, messages=messages)
    raw_text = response.get("message", {}).get("content", "⚠️ Error: No response received.")

    return clean_response(raw_text)

def create_gmail_draft(user_input, body):
    """Fetches and updates the latest Gmail draft, or creates a new one if none exist."""
    if api_resource is None:
        return "⚠️ Error: Gmail API not initialized."

    try:
        # Extract recipient email
        to_email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_input)
        to_email = to_email_match.group(0) if to_email_match else None

        # Extract subject from AI-generated email
        subject_match = re.search(r"^Subject:\s*(.+)", body, re.MULTILINE)
        subject = subject_match.group(1).strip() if subject_match else "No Subject"

        if not to_email:
            return "⚠️ Error: No recipient email found in input."

        # Get sender email
        profile = api_resource.users().getProfile(userId="me").execute()
        sender_email = profile.get("emailAddress", "noreply@example.com")

        # Clean up email body
        cleaned_body = re.sub(r"^Subject:.*\n", "", body, count=1, flags=re.MULTILINE).strip()
        cleaned_body = cleaned_body.replace("[Your Name]", sender_email)

        # Create MIME email
        message = MIMEText(cleaned_body)
        message["to"] = to_email
        message["subject"] = subject
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # ✅ Fetch the latest draft (if any)
        drafts = api_resource.users().drafts().list(userId="me").execute()
        draft_list = drafts.get("drafts", [])

        if draft_list:
            # ✅ Update the most recent draft instead of creating a new one
            latest_draft_id = draft_list[-1]["id"]
            api_resource.users().drafts().update(
                userId="me",
                id=latest_draft_id,
                body={"message": {"raw": encoded_message}}
            ).execute()
            return f"✅ Draft updated for {to_email} with subject: {subject} (Draft ID: {latest_draft_id})"
        else:
            # Create a new draft if none exist
            draft = api_resource.users().drafts().create(
                userId="me",
                body={"message": {"raw": encoded_message}}
            ).execute()
            return f"✅ Draft saved for {to_email} with subject: {subject} (Draft ID: {draft['id']})"

    except Exception as e:
        return f"⚠️ Error saving draft: {str(e)}"

# Dash App Setup
app = Dash(__name__)
app.layout = html.Div([
    dcc.Store(id="store-it", data="[]"),
    html.H1("Email Drafting App (Ollama)"),
    dcc.Textarea(id='llm-request', style={'width': '50%', 'height': '150px'}),
    html.Button('Save Draft', id='btn-draft'),
    html.Div(id='output-space', style={'margin-top': '20px', 'whiteSpace': 'pre-wrap'})
])

@callback(
    [Output('output-space', 'children'),
     Output("store-it", "data")],
    [Input('btn-draft', 'n_clicks')],
    [State('llm-request', 'value'),
     State("store-it", "data")],
    prevent_initial_call=True
)
def draft_email(_, user_input, chat_history_json):
    """Handles email drafting and storing history."""
    try:
        chat_history = json.loads(chat_history_json) if chat_history_json else []

        # Convert chat history format
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["type"] == "human"
            else AIMessage(content=msg["content"])
            for msg in chat_history
        ]

        # Generate AI response for email body
        email_body = process_chat(user_input, chat_history)

        # Save Gmail draft
        draft_result = create_gmail_draft(user_input, email_body)

        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=email_body))

        return f"{draft_result}\n\n{email_body}", json.dumps([
            {"type": "human", "content": user_input},
            {"type": "ai", "content": email_body}
        ])
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}", chat_history_json

if __name__ == "__main__":
    app.run(debug=True)
