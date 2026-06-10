# 📄 DocuQuery - AI PDF Assistant

A beautiful, intuitive AI-powered application to chat with your PDF documents. Built with Streamlit, LangChain, and Groq.

## ✨ Features

- **Drag & Drop PDFs**: Upload your documents with ease
- **Smart Retrieval**: Uses FAISS vector search to find relevant context
- **Powerful AI**: Llama 3 model via Groq for accurate answers
- **Clean UI**: Modern dark theme with smooth interactions
- **Source Citations**: See exactly which parts of your PDFs were used

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A Groq API key (free from [console.groq.com](https://console.groq.com))

### Installation

1. **Clone or download the project**

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**
   - Create a `.env` file in the project root
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

5. **Run the app**
   ```bash
   streamlit run app/app.py
   ```

## 📖 How to Use

1. **Upload PDFs**: Drag & drop your PDF files in the sidebar
2. **Process**: Click "Prepare Documents" to index your files
3. **Chat**: Ask questions about your documents in the chat interface
4. **Explore**: Click on the expanders to see relevant source excerpts

## 🛠️ Tech Stack

- **Streamlit**: Web interface
- **LangChain**: Document processing and LLM orchestration
- **FAISS**: Vector similarity search
- **Groq**: Fast LLM inference with Llama 3
- **Hugging Face**: Text embeddings

## 📝 License

MIT
