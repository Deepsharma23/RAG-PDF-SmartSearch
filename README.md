# DocuQuery

**Chat with your PDFs using AI — grounded answers with source citations.**

DocuQuery is a document assistant that lets you upload PDF files, index their content, and ask natural-language questions. Answers are generated only from your uploaded documents, with expandable source excerpts that show exactly where the information came from.

Built with Streamlit, LangChain, FAISS, and Groq.

---

## Overview

DocuQuery uses Retrieval-Augmented Generation (RAG) to keep answers tied to your files instead of general model knowledge.

1. Upload one or more PDF documents
2. Prepare them into searchable chunks and embeddings
3. Ask questions in a chat-style interface
4. Review answers alongside source document names, page numbers, and excerpts

---

## Features

| Feature | Description |
| --- | --- |
| PDF upload | Upload multiple PDFs (up to 20 MB each) |
| Document preparation | Chunking, embedding, and vector indexing in one step |
| Grounded Q&A | Answers constrained to uploaded document content |
| Source citations | View relevant excerpts with file name and page number |
| Chat interface | Conversation history with a clean, dark workspace UI |
| Fast inference | Llama 3.1 via Groq for low-latency responses |
| Local embeddings | `BAAI/bge-small-en-v1.5` running on CPU |

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Streamlit |
| Orchestration | LangChain |
| Vector search | FAISS |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | Hugging Face (`BAAI/bge-small-en-v1.5`) |
| PDF parsing | PyPDF |

---

## Project Structure

```text
RAG-PDF-SmartSearch/
├── app/
│   ├── app.py          # Streamlit UI and RAG pipeline
│   ├── config.py       # Environment variable loading
│   └── utils.py        # LLM and embedding setup
├── .streamlit/
│   └── config.toml     # Theme and upload size limits
├── requirements.txt
├── .env                # API keys (create locally, not committed)
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- A [Groq API key](https://console.groq.com/) (free tier available)
- Internet access on first run (to download the embedding model)

- ---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Deepsharma23/RAG-PDF-SmartSearch.git
cd RAG-PDF-SmartSearch
```

### 2. Create a virtual environment

**Windows**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Keep your API key private. Never commit `.env` to version control.

---

## Running the App

From the project root:

```bash
streamlit run app/app.py
```

Open the local URL shown in your terminal (default: `http://localhost:8501`).

---

## Usage

### Step 1 — Upload documents

Use the sidebar to browse and upload your PDF files. You can add multiple documents at once.

### Step 2 — Prepare documents

Click **Prepare Documents** to:

- Extract text from each PDF
- Split content into chunks
- Generate embeddings
- Build a FAISS vector index

Wait until the status shows **Ready for questions**.

### Step 3 — Ask questions

Type a question in the chat input at the bottom of the screen. Example prompts:

- *Summarize this document*
- *What are the main conclusions?*
- *Extract important dates and deadlines*

### Step 4 — Review sources

Expand the source sections under each answer to inspect the exact document excerpts used.

Use **New Chat** to clear the conversation while keeping your uploaded documents.

---

## How It Works

```text
PDF Upload
   ↓
Text Extraction (PyPDF)
   ↓
Chunking (1000 chars, 200 overlap)
   ↓
Embeddings (BGE-small-en-v1.5)
   ↓
FAISS Vector Store
   ↓
User Question → Top-k Retrieval (k=5)
   ↓
Groq LLM → Grounded Answer + Sources
```

The assistant is instructed to answer only from retrieved context. If the information is not present in your documents, it responds with:

> I could not find this information in the uploaded documents.

---

## Configuration

| Setting | Location | Default |
| --- | --- | --- |
| Max upload size | `.streamlit/config.toml` | 20 MB |
| LLM model | `app/utils.py` | `llama-3.1-8b-instant` |
| Embedding model | `app/utils.py` | `BAAI/bge-small-en-v1.5` |
| Chunk size / overlap | `app/app.py` | 1000 / 200 |
| Retrieved chunks | `app/app.py` | 5 |

---

## Troubleshooting

**`GROQ_API_KEY not found in .env file`**

- Ensure `.env` exists in the project root (not inside `app/`)
- Confirm the variable name is exactly `GROQ_API_KEY`

**Embedding model download is slow**

- The Hugging Face model is downloaded on first use and cached locally

**Document preparation fails**

- Verify the PDF contains selectable text (scanned images may not work well)
- Check that the file is under 20 MB
- Try re-uploading and clicking **Retry**

**Chat input is disabled**

- Upload at least one PDF and click **Prepare Documents** first

---

## Roadmap

- [ ] Support for additional file types (DOCX, TXT)
- [ ] Persistent vector store across sessions
- [ ] Conversation export
- [ ] Multi-language document support

---

## License

This project is licensed under the MIT License.

---

## Author

**Deep Sharma**

- GitHub: [@Deepsharma23](https://github.com/Deepsharma23)
- Repository: [RAG-PDF-SmartSearch](https://github.com/Deepsharma23/RAG-PDF-SmartSearch)

---

<p align="center">
  <sub>Built with Streamlit · LangChain · FAISS · Groq</sub>
</p>
