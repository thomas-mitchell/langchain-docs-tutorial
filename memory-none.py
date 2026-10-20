# This is an example agent with no memory

# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv('.env')

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

prompt = input("Enter prompt: ")

while prompt:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    print(result["messages"][-1].text)
    prompt = input("Enter prompt: ")