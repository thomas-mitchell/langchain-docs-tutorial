# This is an example agent that accesses JIRA through an MCP server
# (mcp-atlassian: https://github.com/sooperset/mcp-atlassian) instead of the
# langchain JiraToolkit.

# Note that this requires a checkpointer to be created and passed to the agent at creation
# and also the thread_id to be specified in the config when invoking the agent

# pip install -qU langchain "langchain[anthropic]" langchain-mcp-adapters
# The MCP server itself is started on demand with `uvx mcp-atlassian` (needs uv).
import asyncio
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolErrorMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.types import Command
from mcp.client.stdio import get_default_environment

load_dotenv('keyfile.txt')

# LangChain sends traces to LangSmith automatically once tracing is switched on.
# LANGSMITH_API_KEY (and LANGSMITH_ENDPOINT) come from .env.
os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "langsmith-testing")

# The MCP server runs as a separate process and only gets the environment
# variables passed here, so hand it the Jira credentials (under the names
# mcp-atlassian expects) along with the basic system variables it needs to run.
mcp_client = MultiServerMCPClient({
    "jira": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-atlassian"],
        "env": {
            **get_default_environment(),
            "JIRA_URL": os.environ["JIRA_INSTANCE_URL"],
            "JIRA_USERNAME": os.environ["JIRA_USERNAME"],
            "JIRA_API_TOKEN": os.environ["JIRA_API_TOKEN"],
        },
    }
})

# By default a tool exception crashes the whole run. Send the error back to the
# model instead so it can correct itself. Returning None would re-raise instead.
def return_tool_errors(exc, request):
    return f"Error: {type(exc).__name__}: {exc}"


async def main():
    # Keep one MCP session (one server process) open for the whole program.
    # mcp_client.get_tools() would also work, but then every tool call starts
    # a fresh server process, which adds a few seconds per call.
    async with mcp_client.session("jira") as session:
        jira_tools = await load_mcp_tools(session)
        await run_agent(jira_tools)


async def run_agent(jira_tools):
    # MCP tools declare whether they only read data (readOnlyHint). Require
    # approval for every tool that isn't explicitly marked read-only, so new
    # write tools added to the server are gated automatically.
    interrupt_on = {
        t.name: (t.metadata or {}).get("readOnlyHint") is not True
        for t in jira_tools
    }

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=jira_tools,
        system_prompt="You are a helpful assistant that answers questions about JIRA projects and issues.",
        checkpointer=checkpointer, # required for interrupts
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on=interrupt_on),
            ToolErrorMiddleware(return_tool_errors),  # last, so it wraps only the tool call itself
        ],
    )

    # MCP tools are async-only, so the agent has to be run with ainvoke
    config = {"configurable": {"thread_id": "jira-mcp"}}
    prompt = input("Enter prompt: ")

    while prompt:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
        )

        # Keep resuming while the agent is paused waiting for approval
        while "__interrupt__" in result:
            decisions = []
            for request in result["__interrupt__"][0].value["action_requests"]:
                print(f"\nAgent wants to call {request['name']} with:\n{request['args']}")
                if input("Approve? [y/N] ").strip().lower() == "y":
                    decisions.append({"type": "approve"})
                else:
                    decisions.append({"type": "reject", "message": "The user rejected this action."})
            result = await agent.ainvoke(Command(resume={"decisions": decisions}), config)

        print(result["messages"][-1].text)
        prompt = input("Enter prompt: ")


asyncio.run(main())
