# 📄 Contract RAG Assistant

A local, privacy-focused Retrieval Augmented Generation (RAG) system for analyzing contract documents using Ollama, ChromaDB, and Streamlit.

## 🌟 Features

- **100% Local & Private**: All processing happens on your machine, no data sent to external APIs
- **Multi-Document Support**: Upload and query multiple PDF contracts
- **Intelligent Search**: Uses semantic search to find relevant information
- **Chat Interface**: User-friendly web interface powered by Streamlit
- **Source Attribution**: Shows which documents were used to answer questions
- **Multilingual Support**: Supports Turkish and other languages

## 🏗️ Architecture

```
Contract PDFs → Document Processor → Text Chunks → HuggingFace Embeddings
                                                           ↓
User Question → Vector Search (ChromaDB) → Relevant Chunks → Ollama llama3.2 → Answer
```

**Components:**
- **Document Processor**: Loads and chunks PDF files
- **Vector Store**: ChromaDB for storing and searching embeddings
- **RAG Pipeline**: Combines retrieval with Ollama LLM for answers
- **Streamlit UI**: Web interface for interaction

## 📁 Project Structure

```
contract_rag/
├── data/
│   ├── contracts/          # Store your PDF contracts here
│   └── chroma_db/          # Vector database (auto-created)
├── src/
│   ├── document_processor.py   # PDF loading and chunking
│   ├── vector_store.py         # ChromaDB integration
│   └── rag_pipeline.py         # RAG logic with Ollama
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama from: https://ollama.ai

   # Pull llama3.2 model
   ollama pull llama3.2
   ```

### Setup

1. **Clone or download this project**

2. **Install dependencies**
   ```bash
   cd contract_rag
   pip install -r requirements.txt
   ```

3. **Verify Ollama is running**
   ```bash
   ollama list
   # Should show llama3.2
   ```

## 💻 Usage

### Start the Application

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

### Using the Interface

1. **Upload PDFs**
   - Click "Browse files" in the sidebar
   - Select your contract PDF(s)
   - Click "Process Uploaded PDFs"
   - Wait for processing (first time takes 2-5 minutes)

2. **Ask Questions**
   - Type your question in the chat input
   - Example: "What is the contract duration?"
   - View the answer and source documents

3. **Adjust Settings**
   - Use the sidebar slider to change number of source chunks (k)
   - Clear chat history to start fresh

### Example Questions

- "What is the cancellation policy?"
- "What are the payment terms?"
- "Is there an early termination fee?"
- "What are my obligations under this contract?"

## 🔧 Configuration

### Change LLM Model

Edit `app.py` and `src/rag_pipeline.py`:
```python
model_name="llama3.2"  # Change to any Ollama model
```

### Adjust Chunk Size

Edit `src/document_processor.py`:
```python
chunk_size=1000,      # Larger = more context per chunk
chunk_overlap=200     # Overlap between chunks
```

### Change Embedding Model

Edit `src/vector_store.py`:
```python
model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Try other models from HuggingFace
```

## 🛠️ Troubleshooting

### "No module named 'src'"
Make sure you're running from the `contract_rag` directory.

### "Ollama connection error"
Ensure Ollama is running:
```bash
ollama serve
```

### Slow embedding creation
First-time processing is slow (downloads models). Subsequent runs are fast.

### Database errors
Delete `data/chroma_db/` and re-process PDFs:
```bash
rm -rf data/chroma_db
```

## 📊 Performance

- **Initial Setup**: 2-5 minutes (model downloads)
- **PDF Processing**: ~30 seconds per PDF
- **Question Answering**: 3-10 seconds per question
- **Database Loading**: 1-2 seconds

## 🔐 Privacy & Security

- ✅ All data stays on your local machine
- ✅ No external API calls
- ✅ No internet required after setup
- ✅ Your contracts never leave your computer

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Ollama** - Local LLM runtime
- **ChromaDB** - Vector database
- **LangChain** - RAG framework
- **Streamlit** - Web interface
- **HuggingFace** - Embedding models

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📧 Support

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ using 100% local, open-source tools**
