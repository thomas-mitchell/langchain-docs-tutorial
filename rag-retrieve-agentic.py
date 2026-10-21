# Step 2 of 2: answers a question using the Milvus Lite database created by
# rag-embed.py. Run rag-embed.py first; this script doesn't read the PDF.
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
QUESTION = "How much nuyen does a Power focus cost?"
# Joined with spaces: the commas make this a tuple of lines, not one string
SYSTEM_PROMPT = " ".join((
    "You are a helpful assistant that answers questions about Shadowrun 3rd edition",
    "The currency used in Shadowrun 3rd edition is nuyen. If asked about the price",
    "of any item assume the results from the context provided are in Nuyen, unless",
    "specified otherwise (for example some items also cost Karma)",
    "Your response will be printed in a console output, so only use plain text without any special characters.",
    "Use search_documents when you need information from the knowledge base."
))
# The system prompt goes in the prompt as a system message; the model classes
# don't take one. ChatHuggingFace sends both messages to the model's chat template.
# PROMPT = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     ("human", "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n"),
# ])
TOP_K = 8
# Database file created by rag-embed.py
MILVUS_URI = "docling.db"
COLLECTION_NAME = "docling_demo"

# Milvus Lite would otherwise quietly create a new, empty database
if not Path(MILVUS_URI).exists():
    raise SystemExit(f"{MILVUS_URI} not found. Run rag-embed.py first to create it.")

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

# Open the existing collection (no documents added, nothing dropped)
vectorstore = Milvus(
    embedding_function=embedding,
    collection_name=COLLECTION_NAME,
    connection_args={"uri": MILVUS_URI},
)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

def clip_text(text, threshold=100):
    return f"{text[:threshold]}..." if len(text) > threshold else text

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
        Eg if you need to find the price of a cyberdeck search for 'Cyberdeck'
        instead of 'Cyberdeck price' or 'Cyberdeck cost'
    """
    print(f"\nSearching vector DBfor: {query}")
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

# Configure the agent's checkpointer and thread ID (required for checkpointer)
# checkpointer = InMemorySaver()
# config = {"configurable": {"thread_id": "jira-mcp"}}

# Create an agent instead of a retrieval chain
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": QUESTION}
    ],
})


printed_response = console_safe(response["messages"][-1].content)

print()
print(printed_response)

# clipped_answer = clip_text(resp_dict["answer"], threshold=650)
# print(f"Question:\n{resp_dict['input']}\n\nAnswer:\n{clipped_answer}")
# for i, doc in enumerate(resp_dict["context"]):
#     print()
#     print(f"Source {i + 1}:")
#     print(f"  text: {json.dumps(clip_text(doc.page_content, threshold=650))}")
#     for key in doc.metadata:
#         if key != "pk":
#             val = doc.metadata.get(key)
#             clipped_val = clip_text(val) if isinstance(val, str) else val
#             print(f"  {key}: {clipped_val}")
