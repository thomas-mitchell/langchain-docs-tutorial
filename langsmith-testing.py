# This is an example agent with in-memory memory

# Note that this requires a checkpointer to be created and passed to the agent at creation
# and also the thread_id to be specified in the config when invoking the agent

# pip install -qU langchain "langchain[anthropic]"
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

load_dotenv('keyfile.txt')

# LangChain sends traces to LangSmith automatically once tracing is switched on.
# LANGSMITH_API_KEY (and LANGSMITH_ENDPOINT) come from .env.
os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "langsmith-testing")

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

checkpointer = InMemorySaver()
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    checkpointer=checkpointer,
)

prompt = input("Enter prompt: ")

while prompt:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": "memory-demo"}},
    )
    print(result["messages"][-1].text)
    prompt = input("Enter prompt: ")