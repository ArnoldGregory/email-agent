from dash import Dash, html, dcc, callback, Output, Input, State
from langchain_community.agent_toolkits import GmailToolkit
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import find_dotenv, load_dotenv

# Activate API keys
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Connect to Gmail and tools
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials

credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

tools = toolkit.get_tools()

# Use the LLM
instructions = """You are an assistant that creates email drafts."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)

# Define callback and layout for Dash app
app = Dash()
app.layout = [
    dcc.Store(id="store-it", data=[]),
    html.H1("Email Creating App"),
    dcc.Textarea(id='llm-request', style={'width': '30%', 'height': 300}),
    html.Button('Submit', id='btn'),
    html.Div(id='output-space')
]

@callback(
    Output('output-space', 'children'),
    Output("store-it", "data"),
    Input('btn', 'n_clicks'),
    State('llm-request', 'value'),
    State("store-it", "data"),
    prevent_initial_call=True
)
def draft_email(_, user_input, chat_history):
    if len(chat_history) > 0:
        chat_history = loads(chat_history)  # Deserialize the chat_history (convert json to object)
    
    try:
        # Directly process the chat without retry mechanism
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response[0]))  # Assuming response is structured like this

        history = dumps(chat_history)  # Serialize the chat_history (convert the object to json)
        
        return [response[0], html.P()], history  # Only return the final email draft content, not intermediate data

    except Exception as e:
        if "insufficient_quota" in str(e):
            return [html.P("Error: You have exceeded your API quota. Please check your account or try again later.")], chat_history
        return [f"An error occurred: {e}"], chat_history  # General error handling

if __name__ == "__main__":
    app.run(debug=True)
