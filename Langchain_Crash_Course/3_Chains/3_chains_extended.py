from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
import os
from langchain_openai import AzureChatOpenAI

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

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes.")
    ]
)

uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Words Count: {len(x.split())}\n{x}")

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words
response = chain.invoke({"topic": "lawyers", "joke_count": 3})
print(response)