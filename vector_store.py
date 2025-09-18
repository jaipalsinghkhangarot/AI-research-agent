import os
import asyncio
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

CHROMA_PATH = "chroma_db"
#CHROMA_PATH = "chroma_hf"

# Local embedding model (384-dim)
#LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- FIX: Ensure event loop exists ---
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    return asyncio.get_event_loop()

def get_embeddings():
    ensure_event_loop()  # <-- important
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        #transport="rest"   # <-- forces REST instead of gRPC asyncio
    )
    
# def get_embeddings():
#     """Return HuggingFace embeddings."""
#     ensure_event_loop()  # <-- important
#     return HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)

def get_vector_store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )

def store_text(name: str, text: str):
    docs = [Document(page_content=text, metadata={"source": name})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    db = get_vector_store()
    db.add_documents(split_docs)
    db.persist()

def query_vector_store(query: str, k: int = 3):
    db = get_vector_store()
    return db.similarity_search(query, k=k)
