from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_chunk_pdf(pdf_path: str):
    # Step 1: Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {pdf_path}")

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    return chunks

# Test it
if __name__ == "__main__":
    pdf_file = "docs/Global_Mental_Health.pdf"   # ← change to your filename
    chunks = load_and_chunk_pdf(pdf_file)

    # Preview first 2 chunks
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)