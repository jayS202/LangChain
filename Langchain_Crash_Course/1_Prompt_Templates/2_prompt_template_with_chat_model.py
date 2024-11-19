from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

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

# PART 1: Create a ChatPromptTemplate using a template string
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

print("-----Prompt from Template-----")
prompt = prompt_template.invoke({"topic": "cats"})
response = model.invoke(prompt)
print(response.content)


# PART 2: Prompt With Multiple Placeholders
template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)

print("-----Prompt from Multiple Template-----")
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "cat"})
response = model.invoke(prompt)
print(response.content)


# PART 3 (a): Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

print("-----Prompt from Template with tuples-----")
prompt = prompt_template.invoke({'topic':'lawyers','joke_count': '3'})
response = model.invoke(prompt)
print(response.content)


# PART 3 (b)
# This works
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System Messages(Tuple) and Human Messages -----\n")
response = model.invoke(prompt)
print(response.content)


# This does not work
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me {jokes_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({'topic':'lawyers','joke_count': '3'})
print(prompt)