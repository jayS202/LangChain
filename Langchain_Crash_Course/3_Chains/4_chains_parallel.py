from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
import os
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env
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

prompt_templates = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert product reviewer."),
        ("human","List the main features of the product {product_name}."),
    ]
)

def analyze_pros(features):
    pros_templates = ChatPromptTemplate.from_messages(
        [
            ("system","You are an expert product reviewer"),
            ("human","Given these features: {features}, list the pros of these features."),
        ]
    )
    return pros_templates.format_prompt(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human","Given these features: {features}, list the cons of these features."),   
        ]
    )
    return cons_template.format_prompt(features=features)

def combine_pros_cons(pros,cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_templates
    | model
    | StrOutputParser()
    | RunnableParallel(branches = {"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: print("Final Output:\n ", x) or combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name":"MacBook Pro"})
print(result)