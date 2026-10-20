# This is an example agent that can access JIRA

# Note that this requires a checkpointer to be created and passed to the agent at creation
# and also the thread_id to be specified in the config when invoking the agent

# pip install -qU langchain "langchain[anthropic]" langchain-community atlassian-python-api
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolErrorMiddleware
from langgraph.types import Command

load_dotenv('.env')

# LangChain sends traces to LangSmith automatically once tracing is switched on.
# LANGSMITH_API_KEY (and LANGSMITH_ENDPOINT) come from .env.
os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("LANGSMITH_PROJECT", "langsmith-testing")

# JiraAPIWrapper reads JIRA_USERNAME, JIRA_API_TOKEN and JIRA_INSTANCE_URL
# (e.g. https://your-site.atlassian.net) from the environment. It also needs
# JIRA_CLOUD; username + API token auth means Atlassian Cloud.
os.environ.setdefault("JIRA_CLOUD", "true")

jira_api_wrapper = JiraAPIWrapper()
jira_toolkit = JiraToolkit.from_jira_api_wrapper(jira_api_wrapper)
jira_tools = jira_toolkit.get_tools()

# The jql_query tool drops the description before the agent ever sees it.
# This custom tool retrieves full details of a JIRA issue, including description and comments
@tool
def get_issue_details(issue_key: str) -> str:
    """Get the full details of a single JIRA issue, including its description
    and comments. Input is an issue key such as PROJ-123."""
    issue = jira_api_wrapper.jira.issue(
        issue_key, fields="summary,status,assignee,reporter,description,comment"
    )
    f = issue["fields"]
    comments = [
        f"{c['author']['displayName']}: {c['body']}"
        for c in (f.get("comment") or {}).get("comments", [])
    ]
    return (
        f"{issue['key']}: {f['summary']}\n"
        f"Status: {f['status']['name']}\n"
        f"Assignee: {(f.get('assignee') or {}).get('displayName', 'None')}\n"
        f"Reporter: {(f.get('reporter') or {}).get('displayName', 'None')}\n\n"
        f"Description:\n{f.get('description') or '(none)'}\n\n"
        "Comments:\n" + ("\n".join(comments) or "(none)")
    )

jira_tools.append(get_issue_details)

# Jira Cloud assigns issues by account ID, not display name, so the agent needs
# a reliable way to look users up rather than guessing method names.
@tool
def find_jira_user(name_or_email: str) -> str:
    """Find JIRA users by display name or email address. Returns each match's
    display name and accountId. Use the accountId when assigning issues, e.g.
    "assignee": {"accountId": "..."} in create_issue."""
    users = jira_api_wrapper.jira.user_find_by_user_string(query=name_or_email)
    if not users:
        return f"No users found matching {name_or_email!r}"
    return "\n".join(
        f"{u.get('displayName')} (accountId: {u.get('accountId')}, email: {u.get('emailAddress', 'hidden')})"
        for u in users
    )

jira_tools.append(find_jira_user)

# By default a tool exception (e.g. the model calling a Jira method that doesn't
# exist via catch_all_jira_api) crashes the whole run. Send the error back to the
# model instead so it can correct itself. Returning None would re-raise instead.
def return_tool_errors(exc, request):
    return f"Error: {type(exc).__name__}: {exc}"

checkpointer = InMemorySaver()
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=jira_tools,
    system_prompt="You are a helpful assistant that answers questions about JIRA projects and issues.",
    checkpointer=checkpointer, # required for interrupts
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={
            "create_issue": True,
            "create_confluence_page": True,
            "catch_all_jira_api": True,
            "jql_query": False,
            "get_projects": False,
            "get_issue_details": False, # Not necessary, as this is default. Just here for documentation
            "find_jira_user": False,
        }),
        ToolErrorMiddleware(return_tool_errors),  # last, so it wraps only the tool call itself
    ],
)

config = {"configurable": {"thread_id": "jira-reader"}}
prompt = input("Enter prompt: ")

while prompt:
    result = agent.invoke(
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
        result = agent.invoke(Command(resume={"decisions": decisions}), config)


    print(result["messages"][-1].text)
    prompt = input("Enter prompt: ")