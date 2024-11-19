from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime
    
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

tools = [
    Tool(
        name = "Time",
        func = get_current_time,
        description="Useful for when you need to know the current time",
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

llm = AzureChatOpenAI(
    model="Your model name",
    azure_endpoint="Your AzureAI model endpoint",
    api_key="Your AzureAI model Key",
    api_version="Your Azure API version",
    model_version="Your AzureAI model version"
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = agent_executor.invoke({"input": "What time is it?"})

print("Response: ", response) 