from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# Safely set HF_TOKEN — works both locally and on Streamlit Cloud
hf_token = None
if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
elif os.getenv("HF_TOKEN"):
    hf_token = os.getenv("HF_TOKEN")

if hf_token:  # only set if we actually have a value, never set None
    os.environ["HF_TOKEN"] = hf_token

# Path where FAISS index will be saved
FAISS_INDEX_PATH = "faiss_index"

def get_embedding_model():
    """Load the free HuggingFace embedding model"""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="./model_cache"    # ← store model locally in project
    )
    print("Embedding model loaded.")
    return embeddings

def create_vector_store(chunks):
    """Convert chunks to embeddings and store in FAISS"""
    embeddings = get_embedding_model()

    print(f"Creating embeddings for {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save to disk so we don't rebuild every time
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"Vector store saved to '{FAISS_INDEX_PATH}/'")

    return vector_store

def load_vector_store():
    """Load existing FAISS index from disk"""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("No FAISS index found. Run create first.")

    embeddings = get_embedding_model()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded from disk.")
    return vector_store

def get_retriever(vector_store):
    """Convert vector store into a retriever that fetches top 3 chunks"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}   # fetch top 3 most relevant chunks
    )
    return retriever

# Test it
if __name__ == "__main__":
    from loader import load_and_chunk_pdf

    # Load and chunk your PDF
    chunks = load_and_chunk_pdf("docs/Global_Mental_Health.pdf")  # ← same filename as Stage 1

    # Create and save vector store
    vector_store = create_vector_store(chunks)

    # Build retriever and test a query
    retriever = get_retriever(vector_store)

    test_query = "what is this document about?"
    print(f"\nTesting retriever with: '{test_query}'")
    results = retriever.invoke(test_query)

    print(f"\nTop {len(results)} relevant chunks found:\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(doc.page_content[:200])   # preview first 200 chars
        print()