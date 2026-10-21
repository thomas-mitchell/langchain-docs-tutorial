# Step 1 of 2: reads the PDFs (with OCR), splits them into chunks, embeds them and
# saves them to a Milvus Lite database file. This is the slow part, so it only
# needs running when the source documents change. Then use rag-retrieve.py to
# ask questions against the saved database.
import logging
import os
import warnings
from dotenv import load_dotenv
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
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

# All files go into the one collection; each chunk's "source" metadata records
# which file it came from.
FILE_PATHS = [
    "Shadowrun3E-Core.pdf",
    "Shadowrun3E-Cannon-Companion.pdf",
    "Shadowrun3E-Companion.pdf",
    "Shadowrun3E-Magic-in-the-Shadows.pdf",
    "Shadowrun3E-Man-and-Machine.pdf",
    "Shadowrun3E-Matrix.pdf",
    "Shadowrun3E-Rigger-3.pdf",
]
# Embedding model. It also sets the chunk size: HybridChunker reads the model's
# max input length (512 tokens for bge-small-en-v1.5; all-MiniLM-L6-v2 was 256),
# so chunks always fit what the model can embed.
# Must match EMBED_MODEL_ID in rag-retrieve.py.
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EXPORT_TYPE = ExportType.DOC_CHUNKS
# Database file shared with rag-retrieve.py, which reads it
MILVUS_URI = "docling.db"
COLLECTION_NAME = "docling_demo"


loader = DoclingLoader(
    file_path=FILE_PATHS,
    export_type=EXPORT_TYPE,
    chunker=HybridChunker(tokenizer=EMBED_MODEL_ID)
)
docs = loader.load()

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
    # split_text only sees the text, so each split gets just the header
    # metadata. Copy the whole document's metadata (including "source") back
    # onto its splits so they can still be traced to their file.
    splits = []
    for doc in docs:
        for split in splitter.split_text(doc.page_content):
            split.metadata = {**doc.metadata, **split.metadata}
            splits.append(split)
else:
    raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

# drop_old=True replaces the collection on each run, so re-running this script
# rebuilds the database rather than adding duplicate chunks.
vectorstore = Milvus.from_documents(
    documents=splits,
    embedding=embedding,
    collection_name=COLLECTION_NAME,
    connection_args={"uri": MILVUS_URI},
    # Milvus Lite requires a metric; COSINE suits sentence-transformers models
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
    drop_old=True,
)

print(f"Stored {len(splits)} chunks from {', '.join(FILE_PATHS)} in {MILVUS_URI} (collection '{COLLECTION_NAME}')")
