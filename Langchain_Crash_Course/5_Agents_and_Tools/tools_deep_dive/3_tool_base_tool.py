import os
from typing import Type
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")
    
class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description="First number to multiply")
    y: float = Field(description="Second number to multiply")
    
class SimpleSearchTool(BaseTool):
    # Add type annotations for 'name' and 'description'
    name: str = "simple_search"
    description: str = "useful when you need to answer questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput
    
    def _run(self, query: str) -> str:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return f"Search results for: {query}\n\n\n{results}\n"
    
class MultiplyNumbersTool(BaseTool):
    # Add type annotations for 'name' and 'description'
    name: str = "multiply_numbers"
    description: str = "multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs
    
    def _run(self, x: float, y: float) -> str:
        result = x * y
        return f"The product of {x} and {y} is {result}"

tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]


llm = AzureChatOpenAI(
    model="Your model name",
    azure_endpoint="Your AzureAI model endpoint",
    api_key="Your AzureAI model Key",
    api_version="Your Azure API version",
    model_version="Your AzureAI model version"
)

prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the ReAct agent using the create_tool_calling_agent function
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Search for Apple Intelligence"})
print("Response for 'Search for LangChain updates':", response)

response = agent_executor.invoke({"input": "Multiply 10 and 20"})
print("Response for 'Multiply 10 and 20':", response)

