# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langchain[anthropic,mcp]>=1.4.6",
#     "python-dotenv",
# ]
# ///

# This is an example agent that accesses JIRA through an MCP server
# (mcp-atlassian: https://github.com/sooperset/mcp-atlassian) instead of the
# langchain JiraToolkit. It uses langchain's own MCPAdapter (beta) rather than
# the separate langchain-mcp-adapters package.

# Note that this requires a checkpointer to be created and passed to the agent at creation
# and also the thread_id to be specified in the config when invoking the agent

# Run with `uv run jira-mcp-v2.py`. The block at the top of this file gives the
# script its own environment: langchain[mcp] needs MCP SDK 2.x, while the
# project's langchain-mcp-adapters (used by jira-mcp.py) needs MCP SDK 1.x.
# The MCP server itself is started on demand with `uvx mcp-atlassian` (needs uv).
import asyncio
import os
from pathlib import Path
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolErrorMiddleware
from langchain.mcp import MCPAdapter
from fastmcp.client.transports import StdioTransport
from langgraph.types import Command
from mcp.client.stdio import get_default_environment

load_dotenv('keyfile.txt')

# LangChain sends traces to LangSmith automatically once tracing is switched on.
# LANGSMITH_API_KEY (and LANGSMITH_ENDPOINT) come from .env.
os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "langsmith-testing")

system_prompt=(
    "You are a helpful assistant that answers questions about JIRA projects "
    "and issues, and about the weather. Your replies are shown in a plain-text "
    "terminal, so do not use emoji or Markdown formatting."
),

# The MCP server runs as a separate process and only gets the environment
# variables passed here, so hand it the Jira credentials (under the names
# mcp-atlassian expects) along with the basic system variables it needs to run.
# MCPAdapter only treats a plain string as a URL, so a local server has to be
# requested explicitly with a transport like this.
jira_server = StdioTransport(
    command="uvx",
    args=["mcp-atlassian"],
    env={
        **get_default_environment(),
        "JIRA_URL": os.environ["JIRA_INSTANCE_URL"],
        "JIRA_USERNAME": os.environ["JIRA_USERNAME"],
        "JIRA_API_TOKEN": os.environ["JIRA_API_TOKEN"],
    },
)

# Our own weather MCP server. A Path to a .py file tells MCPAdapter to run it
# as a local stdio server with the same Python interpreter as this script.
# weather_server = Path(__file__).parent / "mcp-server-custom-weather.py"
weather_server = Path("mcp-server-custom-weather.py")

# By default a tool exception crashes the whole run. Send the error back to the
# model instead so it can correct itself. Returning None would re-raise instead.
def return_tool_errors(exc, request):
    return f"Error: {type(exc).__name__}: {exc}"


async def main():
    # Keep one MCP connection (one server process) per server open for the whole
    # program. Both servers start on entering this block and stop on leaving it.
    async with (
        MCPAdapter(jira_server) as jira_adapter,
        MCPAdapter(weather_server) as weather_adapter,
    ):
        jira_tools = await jira_adapter.list_tools()
        weather_tools = await weather_adapter.list_tools()
        await run_agent(jira_tools + weather_tools)


async def run_agent(tools):
    # MCP tools declare whether they only read data (read-only hint). Require
    # approval for every tool that isn't explicitly marked read-only, so new
    # write tools added to the server are gated automatically.
    # MCPAdapter keeps the hints under metadata["mcp"]["tool"]["annotations"],
    # with snake_case names (MCP SDK 2.x), e.g. read_only_hint.
    def is_read_only(t):
        mcp_tool = (t.metadata or {}).get("mcp", {}).get("tool", {})
        return mcp_tool.get("annotations", {}).get("read_only_hint") is True

    interrupt_on = {t.name: not is_read_only(t) for t in tools}

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=tools,
        system_prompt=system_prompt,
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

        # Keep resuming while the agent is paused waiting for approval.
        # MCPAdapter also uses interrupts when a server asks the user a question
        # mid-call (elicitation); mcp-atlassian doesn't, so only approval
        # requests are handled here.
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
