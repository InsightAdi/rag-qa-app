import streamlit as st
from loader import load_and_chunk_pdf
from retriever import create_vector_store, get_retriever
from chain import build_rag_chain
import tempfile
import os

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="centered"
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f1117;
        color: #ffffff;
    }

    /* Chat message bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 15px;
        line-height: 1.5;
    }

    .assistant-bubble {
        background: #1e1e2e;
        color: #e2e8f0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        border: 1px solid #2d2d3d;
        font-size: 15px;
        line-height: 1.5;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .app-subtitle {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* Upload area */
    .upload-card {
        background: #1e1e2e;
        border: 2px dashed #2d2d3d;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* Status badge */
    .status-ready {
        background: #052e16;
        color: #4ade80;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        border: 1px solid #166534;
        display: inline-block;
    }

    .status-waiting {
        background: #1c1917;
        color: #fb923c;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        border: 1px solid #9a3412;
        display: inline-block;
    }

    /* Stats row */
    .stats-box {
        background: #1e1e2e;
        border: 1px solid #2d2d3d;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stats-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb;
    }

    .stats-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d0d1a;
        border-right: 1px solid #1e1e2e;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #1e1e2e;
        border: 1px solid #2d2d3d;
        border-radius: 12px;
    }

    /* Divider */
    hr {
        border-color: #2d2d3d;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "page_count" not in st.session_state:
    st.session_state.page_count = 0

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DocMind AI")
    st.markdown("<p style='color:#64748b; font-size:13px'>Powered by LangChain + Groq</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        help="Supports any PDF — research papers, reports, books"
    )

    if uploaded_file is not None:
        st.markdown(f"<p style='color:#94a3b8; font-size:13px'>📄 {uploaded_file.name}</p>", unsafe_allow_html=True)

        if st.button("⚡ Process PDF", type="primary", use_container_width=True):
            with st.spinner("Reading and indexing your document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    chunks = load_and_chunk_pdf(tmp_path)
                    page_count = len(set([c.metadata.get("page", 0) for c in chunks]))

                    vector_store = create_vector_store(chunks)
                    retriever = get_retriever(vector_store)
                    st.session_state.rag_chain = build_rag_chain(retriever)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.page_count = page_count
                    st.session_state.chat_history = []

                    os.unlink(tmp_path)
                    st.success("Document ready!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    # Status indicator
    if st.session_state.pdf_processed:
        st.markdown(f"<span class='status-ready'>✅ Ready to chat</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Document stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='stats-box'>
                <div class='stats-number'>{st.session_state.page_count}</div>
                <div class='stats-label'>Pages</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='stats-box'>
                <div class='stats-number'>{st.session_state.chunk_count}</div>
                <div class='stats-label'>Chunks</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b; font-size:12px'>📄 {st.session_state.pdf_name}</p>", unsafe_allow_html=True)

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("<span class='status-waiting'>⏳ No document loaded</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='color:#374151; font-size:11px; text-align:center'>Built with LangChain · FAISS · Groq · Streamlit</p>", unsafe_allow_html=True)

# ─── Main Area ───────────────────────────────────────────────

# Header
st.markdown("""
<div class='app-header'>
    <div class='app-title'>🧠 DocMind AI</div>
    <div class='app-subtitle'>Upload any PDF and have an intelligent conversation with it</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Empty state — no PDF yet
if not st.session_state.pdf_processed:
    st.markdown("""
    <div class='upload-card'>
        <h3 style='color:#94a3b8'>👈 Start by uploading a PDF</h3>
        <p style='color:#64748b'>Use the sidebar to upload your document.<br>
        Supports research papers, reports, books, manuals — any PDF.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stats-box' style='padding:1.2rem'>
            <div style='font-size:1.8rem'>⚡</div>
            <div style='color:#e2e8f0; font-weight:600; margin-top:8px'>Fast</div>
            <div style='color:#64748b; font-size:12px; margin-top:4px'>Powered by Groq LPU for instant answers</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stats-box' style='padding:1.2rem'>
            <div style='font-size:1.8rem'>🎯</div>
            <div style='color:#e2e8f0; font-weight:600; margin-top:8px'>Accurate</div>
            <div style='color:#64748b; font-size:12px; margin-top:4px'>Answers grounded only in your document</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stats-box' style='padding:1.2rem'>
            <div style='font-size:1.8rem'>🔒</div>
            <div style='color:#e2e8f0; font-weight:600; margin-top:8px'>Private</div>
            <div style='color:#64748b; font-size:12px; margin-top:4px'>Your document is never stored permanently</div>
        </div>""", unsafe_allow_html=True)

# Chat history display
else:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align:center; padding: 2rem 0; color:#64748b'>
            <div style='font-size:2.5rem'>💬</div>
            <p>Your document is ready! Ask me anything about it.</p>
            <p style='font-size:13px'>Try: "Summarize this document" or "What are the key findings?"</p>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div class='user-bubble'>🧑 {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'>🧠 {message['content']}</div>", unsafe_allow_html=True)

# Chat input
if question := st.chat_input("Ask anything about your document..."):
    if not st.session_state.pdf_processed:
        st.warning("Please upload and process a PDF first using the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.markdown(f"<div class='user-bubble'>🧑 {question}</div>", unsafe_allow_html=True)

        with st.spinner(""):
            answer = st.session_state.rag_chain.invoke(question)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.markdown(f"<div class='assistant-bubble'>🧠 {answer}</div>", unsafe_allow_html=True)