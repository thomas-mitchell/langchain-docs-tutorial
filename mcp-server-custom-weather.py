# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp>=4.0.11",
# ]
# ///

# A minimal custom MCP server with one tool, get_weather.
# It talks over stdio, so it is normally started by an MCP client (see
# jira-mcp-v2.py) rather than run by hand. `uv run mcp-server-custom-weather.py`
# starts it standalone, waiting for MCP messages on stdin.
from fastmcp import FastMCP

mcp = FastMCP("weather")


# Marked read-only so clients (like the approval middleware in jira-mcp-v2.py)
# know it doesn't change anything.
@mcp.tool(annotations={"read_only_hint": True})
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return "Sunny" if city.strip().lower() == "sydney" else "Rainy"


if __name__ == "__main__":
    # The banner goes to stderr, which would clutter the client's console
    mcp.run(show_banner=False)
