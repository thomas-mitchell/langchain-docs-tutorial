# Step 2 of 2: answers a question using the Milvus Lite database created by
# rag-embed.py. Run rag-embed.py first; this script doesn't read the PDF.
import os
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
    "specified otherwise (for example some items also cost Karma)"
))
# The system prompt goes in the prompt as a system message; the model classes
# don't take one. ChatHuggingFace sends both messages to the model's chat template.
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n"),
])
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

def clip_text(text, threshold=100):
    return f"{text[:threshold]}..." if len(text) > threshold else text

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
# Inference Providers serve instruct models as chat ("conversational") only,
# so wrap the endpoint in ChatHuggingFace rather than using text-generation.
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=GEN_MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",
    )
)

question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
resp_dict = rag_chain.invoke({"input": QUESTION})

clipped_answer = clip_text(resp_dict["answer"], threshold=650)
print(f"Question:\n{resp_dict['input']}\n\nAnswer:\n{clipped_answer}")
for i, doc in enumerate(resp_dict["context"]):
    print()
    print(f"Source {i + 1}:")
    print(f"  text: {json.dumps(clip_text(doc.page_content, threshold=650))}")
    for key in doc.metadata:
        if key != "pk":
            val = doc.metadata.get(key)
            clipped_val = clip_text(val) if isinstance(val, str) else val
            print(f"  {key}: {clipped_val}")
