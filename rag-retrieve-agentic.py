# Step 2 of 2: answers a question using the Milvus Lite database created by
# rag-embed.py. Run rag-embed.py first; this script doesn't read the PDF.
import asyncio
import os
import sys
import json
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# Hugging Face caches the embedding model using symlinks, which Windows only
# allows in Developer Mode. Without them the cache still works (it just copies
# files), so hide the warning. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv('keyfile.txt')

HF_TOKEN = os.environ["HF_TOKEN"]
# Must be the same model rag-embed.py used, or the question's vector won't be
# comparable with the stored ones.
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
# Must be served by a Hugging Face Inference Provider; Mixtral-8x7B no longer is.
# Check a model page's "Inference Providers" panel before switching models.
GEN_MODEL_ID = "openai/gpt-oss-20b"
SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about Shadowrun 3rd edition.
Your response will be printed in a console output, so only use plain text without any special characters.
Use search_documents when you need information from the knowledge base.

When answering using retrieved documents:
- Cite each of your sources separately at the end of your answer.
- ALWAYS include the page numbers for your sources.
- Only cite sources actually returned by search_documents and were relevant to answering the query.
"""
TOP_K = 8
# Database file created by rag-embed.py
MILVUS_URI = "docling.db"
COLLECTION_NAME = "docling_demo"

# Milvus Lite would otherwise quietly create a new, empty database
if not Path(MILVUS_URI).exists():
    raise SystemExit(f"{MILVUS_URI} not found. Run rag-embed.py first to create it.")

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

# Open the existing collection (no documents added, nothing dropped).
# pymilvus pings its gRPC connection every 10s even while idle, but the local
# Milvus Lite server only allows idle pings every 5 minutes. Between questions
# it answers with GOAWAY "too_many_pings" and gRPC logs an error, so only ping
# while a request is actually in progress.
vectorstore = Milvus(
    embedding_function=embedding,
    collection_name=COLLECTION_NAME,
    connection_args={
        "uri": MILVUS_URI,
        "grpc_options": {"grpc.keepalive_permit_without_calls": False},
    },
)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# LLMs (gpt-oss especially) like typographic characters that the PowerShell
# console font or code page can't display, so swap them for plain ASCII.
CONSOLE_SAFE_CHARS = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "--",
    "―": "--", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "⁠": "", "﻿": "",
    "…": "...", "•": "*", "×": "x",
})

def console_safe(text):
    return text.translate(CONSOLE_SAFE_CHARS)

# Anything left that the console's code page can't encode becomes "?" rather
# than raising UnicodeEncodeError (e.g. when output is redirected to a file).
sys.stdout.reconfigure(errors="replace")

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# Docling chunks have no "page" metadata key. Each chunk lists the document
# items it was built from under dl_meta, and each item records the page(s) it
# came from (already 1-based). A chunk can span more than one page.
def get_page_numbers(doc):
    doc_items = doc.metadata.get("dl_meta", {}).get("doc_items", [])
    return sorted({prov["page_no"] for item in doc_items for prov in item.get("prov", [])})

# Expose retrieval as a tool
@tool
def search_documents(query: str) -> str:
    """
        Search the Shadowrun 3rd edition knowledge base for relevant information.
        All information in this knowledge base is Shadowrun 3rd edition specific,
        so do not include terms like 'Shadowrun 3rd edition' in your search queries.

        A lot of data in this knowledge base will be available in tables, including
        item prices. For that reason when searching for data especially item prices
        first try simply searching for the item name without additional context.
        All prices will be in Nuyen (Shadowruns main currency) unless specified otherwise
        (for example some items also cost Karma).
        Eg if you need to find the price of a cyberdeck search for 'Cyberdeck'
        instead of 'Cyberdeck price' or 'Cyberdeck cost'
    """
    print(f"\nSearching Vector DB: {query}") # debugging info
    docs = retriever.invoke(query)
    # return "\n\n".join(doc.page_content for doc in docs)
    results = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        pages = get_page_numbers(doc)

        reference = f"Source {i}: {source}"

        if pages:
            label = "page" if len(pages) == 1 else "pages"
            reference += f", {label} {', '.join(map(str, pages))}"

        results.append(
            f"{reference}\n{doc.page_content}"
        )

        

    return "\n\n".join(results)

# Configure the agent's checkpointer and thread ID (required for checkpointer)
checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "jira-mcp"}}

# Create an agent instead of a retrieval chain
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

async def begin_prompting():
    user_prompt = input("Enter your question: ")

    while user_prompt.strip():
        response = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            },
            config=config,
        )

        printed_response = console_safe(response["messages"][-1].content)
        print()
        print(printed_response)
        
        user_prompt = input("Enter your question: ")

asyncio.run(begin_prompting())
