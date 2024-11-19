# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

# Import necessary libraries
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_structured_chat_agent, create_react_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool, Tool
from langchain_openai import AzureChatOpenAI

def greet_user(name: str)->str:
    """Greets the user by name"""
    return f"Hello {name}"

def reverse_string(text: str)->str:
    """Reverses the given string"""
    return text[::-1]

def concatenate_strings(a: str, b: str)->str:
    """Concatenates two strings"""
    return a + b

class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First String")
    b: str = Field(description="Second String")
    
tools = [
    Tool(
        name = "GreetUser",
        func = greet_user,
        description = "Greets the user by name.",
    ),
    
    Tool(
        name = "ReverseString",
        func = reverse_string,
        description = "Reverses the given string"
    ),
    
    StructuredTool.from_function(
        name="ConcatenateStrings",
        func=concatenate_strings,
        description="Concatenates two strings",
        args_schema=ConcatenateStringsArgs
    )
]

llm = AzureChatOpenAI(
    model="Your model name",
    azure_endpoint="Your AzureAI model endpoint",
    api_key="Your AzureAI model Key",
    api_version="Your Azure API version",
    model_version="Your AzureAI model version"
)

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)

