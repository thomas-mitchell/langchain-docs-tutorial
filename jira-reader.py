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

load_dotenv('.env')

# LangChain sends traces to LangSmith automatically once tracing is switched on.
# LANGSMITH_API_KEY (and LANGSMITH_ENDPOINT) come from .env.
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ.setdefault("LANGSMITH_PROJECT", "langsmith-testing")

# JiraAPIWrapper reads JIRA_USERNAME, JIRA_API_TOKEN and JIRA_INSTANCE_URL
# (e.g. https://your-site.atlassian.net) from the environment. It also needs
# JIRA_CLOUD; username + API token auth means Atlassian Cloud.
os.environ.setdefault("JIRA_CLOUD", "true")

jira_api_wrapper = JiraAPIWrapper()
jira_toolkit = JiraToolkit.from_jira_api_wrapper(jira_api_wrapper)

# Only give the agent the read-only tools. The toolkit also has create_issue,
# create_confluence_page and catch_all_jira_api (which can call any Jira API
# method, including deletes).
read_only_tools = {"jql_query", "get_projects"}
jira_tools = [t for t in jira_toolkit.get_tools() if t.name in read_only_tools]

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

checkpointer = InMemorySaver()
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=jira_tools,
    system_prompt="You are a helpful assistant that answers questions about JIRA projects and issues.",
    checkpointer=checkpointer,
)

prompt = input("Enter prompt: ")

while prompt:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": "jira-reader"}},
    )
    print(result["messages"][-1].text)
    prompt = input("Enter prompt: ")