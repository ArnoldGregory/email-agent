import ollama
import json
import base64
from dash import Dash, html, dcc, callback, Output, Input, State
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import find_dotenv, load_dotenv
from email.mime.text import MIMEText

# Load API keys
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Connect to Gmail API
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# AI Model Configuration
MODEL_NAME = "deepseek-r1:1.5b"

def generate_response(user_input, chat_history):
    """Generates an AI response using DeepSeek-R1 (Ollama)."""
    messages = [{"role": "system", "content": "You are an AI assistant that drafts professional emails."}]

    for msg in chat_history:
        role = "user" if msg["type"] == "human" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": user_input})
    
    response = ollama.chat(model=MODEL_NAME, messages=messages)
    
    return response.get("message", {}).get("content", "⚠️ Error: No response received.")

def create_draft(to_email, subject, body):
    """Saves a draft email in Gmail."""
    try:
        message = MIMEText(body)
        message["to"] = to_email
        message["subject"] = subject
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        draft = api_resource.users().drafts().create(
            userId="me", body={"message": {"raw": encoded_message}}
        ).execute()

        return f"✅ Draft saved! Draft ID: {draft['id']}"
    
    except Exception as e:
        return f"⚠️ Error saving draft: {str(e)}"

# Dash App Setup
app = Dash(__name__)
app.layout = html.Div([
    dcc.Store(id="store-it", data="[]"),  
    html.H1("DeepSeek Email Drafting App"),
    dcc.Textarea(id='llm-request', style={'width': '50%', 'height': '150px'}),
    html.Button('Submit', id='btn'),
    html.Div(id='output-space', style={'margin-top': '20px', 'whiteSpace': 'pre-wrap'})
])

@callback(
    [Output('output-space', 'children'), Output("store-it", "data")],
    [Input('btn', 'n_clicks')],
    [State('llm-request', 'value'), State("store-it", "data")],
    prevent_initial_call=True
)
def draft_email(_, user_input, chat_history_json):
    """Handles email drafting and saving."""
    try:
        chat_history = json.loads(chat_history_json) if chat_history_json else []
        email_body = generate_response(user_input, chat_history)

        # Generate a dynamic subject from the first sentence of the AI response
        subject = email_body.split(".")[0][:50] + "..." if "." in email_body else "Follow-up Email"

        chat_history.append({"type": "human", "content": user_input})
        chat_history.append({"type": "ai", "content": email_body})

        # Save draft
        draft_result = create_draft(to_email="mike.castles@email.com", subject=subject, body=email_body)

        return f"{draft_result}\n\n{email_body}", json.dumps(chat_history)

    except Exception as e:
        return f"⚠️ Error: {str(e)}", chat_history_json

if __name__ == "__main__":
    app.run(debug=True)
