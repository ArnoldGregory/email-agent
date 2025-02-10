import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dash import Dash, html, dcc, callback, Output, Input, State
import json

# Model Name
MODEL_NAME = "deepseek-ai/DeepSeek-R1"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    quantization_config={}  # Add this argument
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)

# Function to generate AI response
def generate_response(user_input, chat_history):
    messages = "You are a helpful assistant.\n"
    for msg in chat_history:
        role = "User" if msg["type"] == "human" else "AI"
        messages += f"{role}: {msg['content']}\n"
    
    messages += f"User: {user_input}\nAI:"
    
    inputs = tokenizer(messages, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=300)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("AI:")[-1]
    
    return response.strip()

# Initialize Dash App
app = Dash()
app.layout = html.Div([
    dcc.Store(id="chat-history", data="[]"),  # Store chat history
    html.H1("DeepSeek Chatbot"),
    dcc.Textarea(id='user-input', style={'width': '50%', 'height': '150px'}),
    html.Button('Submit', id='submit-btn'),
    html.Div(id='response-output', style={'margin-top': '20px'})
])

# Callback for handling chat responses
@callback(
    [Output('response-output', 'children'),
     Output("chat-history", "data")],
    [Input('submit-btn', 'n_clicks')],
    [State('user-input', 'value'),
     State("chat-history", "data")],
    prevent_initial_call=True
)
def chat(_, user_input, chat_history_json):
    try:
        chat_history = json.loads(chat_history_json) if chat_history_json else []
        response = generate_response(user_input, chat_history)
        
        chat_history.append({"type": "human", "content": user_input})
        chat_history.append({"type": "ai", "content": response})

        return response, json.dumps(chat_history)
    
    except Exception as e:
        return f"Error: {str(e)}", chat_history_json

if __name__ == "__main__":
    app.run(debug=True)
