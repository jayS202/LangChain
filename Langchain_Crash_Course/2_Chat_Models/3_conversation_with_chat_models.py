from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
import os

load_dotenv()

GPT4_API_KEY=os.getenv('GPT4_API_KEY')
GPT4_API_BASE=os.getenv('GPT4_API_BASE')
GPT4_API_VERSION=os.getenv('GPT4_API_VERSION')
GPT4_API_DEPLOYMENT_NAME=os.getenv('GPT4_API_DEPLOYMENT_NAME')

os.environ["AZURE_OPENAI_API_KEY"] = GPT4_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = GPT4_API_BASE


model = AzureChatOpenAI(
    azure_deployment=GPT4_API_DEPLOYMENT_NAME,
    api_version=GPT4_API_VERSION, 
    temperature=0,
    max_tokens=None,
    timeout=None
)

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    question = input("You: ")
    if(question == "exit"):
        break
    chat_history.append(HumanMessage(content=question))
    
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")
    
    
print("---------- Message History ------------")
print(chat_history)