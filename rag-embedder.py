import logging
import os
import warnings
import json
from dotenv import load_dotenv
from pathlib import Path
from tempfile import mkdtemp
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
# from langchain_docling import DoclingLoader
from langchain_docling.loader import DoclingLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# Hugging Face caches the Docling models using symlinks, which Windows only
# allows in Developer Mode. Without them the cache still works (it just copies
# files), so hide the warning. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv('keyfile.txt')

# langchain-docling registers a Docling plugin (for describing pictures with a
# LangChain model) that Docling refuses to load unless external plugins are
# allowed. It isn't used here, so hide the warning Docling logs every time.
logging.getLogger("docling.models.factories.base_factory").setLevel(logging.ERROR)

# The OCR engine logs every model it loads; only show its warnings and errors.
# It resets its own log level each time it starts, so a filter is used instead
# of setLevel.
logging.getLogger("RapidOCR").addFilter(lambda record: record.levelno >= logging.WARNING)

# Harmless performance note from PyTorch inside the OCR model
warnings.filterwarnings("ignore", message="Using padding='same'")

FILE_PATH = "Shadowrun3E-Core.pdf"
HF_TOKEN = os.environ["HF_TOKEN"]
# Embedding model. It also sets the chunk size: HybridChunker reads the model's
# max input length (512 tokens for bge-small-en-v1.5; all-MiniLM-L6-v2 was 256),
# so chunks always fit what the model can embed.
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
# Must be served by a Hugging Face Inference Provider; Mixtral-8x7B no longer is.
# Check a model page's "Inference Providers" panel before switching models.
GEN_MODEL_ID = "openai/gpt-oss-20b"
EXPORT_TYPE = ExportType.DOC_CHUNKS
QUESTION = "How much does a Power focus cost?"
PROMPT = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n",
)
TOP_K = 3
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")


loader = DoclingLoader(
    file_path=FILE_PATH,
    export_type=EXPORT_TYPE,
    chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    convert_kwargs={"page_range": (312, 316)}
)
# convert_kwargs={"page_range": (51, 54)}

docs = loader.load()

# Debug output
# for d in docs[:3]:
#     print(f"- {d.page_content=}")

if EXPORT_TYPE == ExportType.DOC_CHUNKS:
    splits = docs
elif EXPORT_TYPE == ExportType.MARKDOWN:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ],
    )
    splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
else:
    raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

# Debug output
# for d in splits[:3]:
#     print(f"- {d.page_content=}")
# print("...")

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

milvus_uri = str(Path(mkdtemp()) / "docling.db")  # or set as needed
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=embedding,
    collection_name="docling_demo",
    connection_args={"uri": milvus_uri},
    # Milvus Lite requires a metric; COSINE suits sentence-transformers models
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
    drop_old=True,
)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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

def clip_text(text, threshold=100):
    return f"{text[:threshold]}..." if len(text) > threshold else text

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