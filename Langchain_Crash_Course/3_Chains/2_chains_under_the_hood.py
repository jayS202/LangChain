from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
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

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first = format_prompt, middle=[invoke_model], last=parse_output)

result = chain.invoke({"topic": "lawyers", "joke_count": 3})
print(result)