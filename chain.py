from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()  # works locally
# Override with Streamlit Cloud secrets if available
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def get_llm():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
    return llm

def build_rag_chain(retriever):
    """Build the full RAG pipeline"""

    prompt_template = """
    You are a helpful assistant. Answer the question using ONLY the 
    context provided below. If the answer is not in the context, 
    say "I don't have enough information to answer this."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    llm = get_llm()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question: str):
    """Send a question through the RAG chain"""
    print(f"\nQuestion: {question}")
    print("Thinking...\n")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer

# Test it
if __name__ == "__main__":
    from retriever import load_vector_store, get_retriever

    print("Loading vector store...")
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)

    print("Building RAG chain...")
    rag_chain = build_rag_chain(retriever)

    questions = [
        "what is this document about?",
        "summarize the main points",
        "what are the key topics covered?"
    ]

    for question in questions:
        ask_question(rag_chain, question)
        print("-" * 50)