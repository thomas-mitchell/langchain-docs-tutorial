# This is an example agent with persisted database memory

# Note that this requires a Sqlite checkpointer to be created and passed to the agent at
# creation and also the thread_id to be specified in the config when invoking the agent

# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools import tool

load_dotenv('.env')

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

with SqliteSaver.from_conn_string("memory-demo.db") as checkpointer:
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