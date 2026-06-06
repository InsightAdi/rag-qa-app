# 🧠 DocMind AI — Chat with Any PDF

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-orange?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A production-grade Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and have an intelligent conversation with it.**

[🚀 Live Demo](https://rag-app-app.streamlit.app/) · [📂 GitHub Repo](https://github.com/InsightAdi/rag-qa-app) · [🐛 Report Bug](https://github.com/InsightAdi/rag-qa-app/issues)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Usage](#-usage)
- [Architecture Deep Dive](#-architecture-deep-dive)
- [Real-World Use Cases](#-real-world-use-cases)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## 📖 Overview

**DocMind AI** is a full-stack AI application built on the **RAG (Retrieval-Augmented Generation)** architecture. Instead of relying on an LLM's training data, it retrieves relevant information directly from your uploaded document and generates grounded, accurate answers.

### ✨ Key Features

- 📄 **Upload any PDF** — research papers, reports, books, manuals
- 💬 **Conversational Q&A** — ask questions in natural language
- ⚡ **Fast responses** — powered by Groq's LPU inference engine
- 🎯 **Grounded answers** — LLM only answers from your document, no hallucinations
- 🔒 **Privacy first** — documents are never stored permanently
- 🌐 **Deployed & shareable** — live on Streamlit Cloud, accessible via URL

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PDF Upload                                                 │
│      │                                                      │
│      ▼                                                      │
│  [Document Loader]  ──── PyPDF reads pages                  │
│      │                                                      │
│      ▼                                                      │
│  [Text Splitter]  ────── 500-char chunks, 50 overlap        │
│      │                                                      │
│      ▼                                                      │
│  [Embedding Model] ───── all-MiniLM-L6-v2 (HuggingFace)    │
│      │                                                      │
│      ▼                                                      │
│  [FAISS Vector Store] ── Stores & indexes all vectors       │
│                                                             │
│  User Question                                              │
│      │                                                      │
│      ▼                                                      │
│  [Retriever] ─────────── Top-3 similar chunks fetched       │
│      │                                                      │
│      ▼                                                      │
│  [Prompt Template] ────── Question + Context combined       │
│      │                                                      │
│      ▼                                                      │
│  [Groq LLaMA 3.1] ─────── Generates grounded answer        │
│      │                                                      │
│      ▼                                                      │
│  Final Answer shown in chat UI                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq + LLaMA 3.1 8B | Fast inference, answer generation |
| **RAG Framework** | LangChain | Pipeline orchestration |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Text → vector conversion |
| **Vector Store** | FAISS (Meta) | Similarity search over chunks |
| **Document Loader** | PyPDF | PDF parsing and extraction |
| **Frontend** | Streamlit | Chat UI and file upload |
| **Deployment** | Streamlit Cloud | Free hosting, shareable URL |
| **Language** | Python 3.9+ | Core language |

---

## 📁 Project Structure

```
rag-qa-app/
│
├── app.py              # Streamlit UI — chat interface, file upload, session state
├── loader.py           # PDF loading and recursive text chunking
├── retriever.py        # HuggingFace embeddings + FAISS vector store
├── chain.py            # LangChain RAG chain + Groq LLM integration
│
├── requirements.txt    # All Python dependencies
├── .gitignore          # Excludes venv, .env, model cache, FAISS index
└── README.md           # This file
```

### What Each File Does

**`loader.py`**
- Loads PDF using `PyPDFLoader`
- Splits text into 500-character chunks with 50-character overlap using `RecursiveCharacterTextSplitter`
- Returns a list of `Document` objects ready for embedding

**`retriever.py`**
- Loads `all-MiniLM-L6-v2` embedding model from HuggingFace
- Converts document chunks into vector embeddings
- Stores and retrieves vectors using FAISS
- Saves the index to disk so PDFs don't need reprocessing

**`chain.py`**
- Connects to Groq API (LLaMA 3.1 8B Instant)
- Defines a prompt template that constrains answers to document context
- Builds a LangChain LCEL pipeline: `retriever → prompt → LLM → output parser`

**`app.py`**
- Streamlit chat interface with dark theme
- Sidebar for PDF upload and processing
- Session state management for chat history and RAG chain persistence
- Displays document stats (pages, chunks) after processing

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- A free [Groq API key](https://console.groq.com)
- A free [HuggingFace token](https://huggingface.co/settings/tokens)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/rag-qa-app.git
cd rag-qa-app
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env            # or create .env manually
```

Add your keys to `.env`:
```
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_hf_token_here
```

**5. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔐 Environment Variables

| Variable | Required | Where to Get |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) → API Keys |
| `HF_TOKEN` | ⚠️ Optional | [huggingface.co](https://huggingface.co/settings/tokens) → New Token |

> `HF_TOKEN` is optional but recommended to avoid HuggingFace rate limits on the embedding model download.

---

## 💡 Usage

1. **Open the app** at `http://localhost:8501` or the live URL
2. **Upload a PDF** using the sidebar file uploader
3. **Click "Process PDF"** — wait for the green success message
4. **Ask questions** in the chat input at the bottom
5. **Get grounded answers** based only on your document content

### Example Questions to Try

```
"What is this document about?"
"Summarize the key findings"
"What are the main recommendations?"
"Explain the methodology used"
"What conclusions does the author draw?"
```

---

## 🏗️ Architecture Deep Dive

### Why RAG Over Fine-Tuning?

| Approach | Cost | Time | Updatable |
|---|---|---|---|
| Fine-tuning | High 💰 | Days | No — retrain needed |
| RAG | Free ✅ | Minutes | Yes — swap the PDF |

RAG is the industry standard for document Q&A because it's cheaper, faster, and the knowledge source can be updated instantly.

### Why FAISS Over Other Vector DBs?

- **Free & local** — no external service needed
- **Fast** — Meta's optimized similarity search
- **Persistent** — index saved to disk, no reprocessing
- Production alternative: **Pinecone** or **ChromaDB** for multi-user scale

### Why Groq?

- **Free tier** with generous limits
- **10x faster** than standard OpenAI API due to LPU hardware
- Drop-in replacement — same LangChain interface as OpenAI

### Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,      # ~100-150 words per chunk
    chunk_overlap=50     # prevents context loss at boundaries
)
```

Overlap ensures that sentences split across chunk boundaries are still retrievable — critical for accurate answers.

---

## 🌍 Real-World Use Cases

| Domain | Application |
|---|---|
| 🏥 Healthcare | Query medical research papers or patient reports |
| ⚖️ Legal | Chat with contracts, case files, or legislation |
| 📈 Finance | Ask questions over earnings reports or filings |
| 🎓 Education | Students chatting with textbooks or lecture notes |
| 🏢 Enterprise | Internal knowledge base / employee handbook Q&A |
| 🛠️ Engineering | Query technical documentation or API references |

---

## 🗺️ Roadmap

- [x] Single PDF upload and Q&A
- [x] FAISS vector store with persistence
- [x] Streamlit chat UI with dark theme
- [x] Deployed on Streamlit Cloud
- [ ] Multi-PDF support
- [ ] Conversation memory (multi-turn context)
- [ ] FastAPI backend for REST API access
- [ ] Streaming responses
- [ ] Support for DOCX, TXT, CSV files
- [ ] Switch to Pinecone for multi-user scale

---

## 👨‍💻 Author

**Aditya Yadav**
AI/ML Engineer | Data Analyst

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/aditya-y-a757131ba/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/InsightAdi)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:adityaydv1203@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ using LangChain · FAISS · Groq · Streamlit
</div>
