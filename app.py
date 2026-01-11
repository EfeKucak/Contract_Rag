"""
Streamlit Web Interface for Contract RAG System

Main Features:
1. PDF Upload: Drag & drop PDFs to add to database
2. Chat Interface: Ask questions about contracts
3. Sidebar: Upload PDFs, view stats, adjust settings
4. Source Display: Show which documents were used for answers
5. Chat History: Remember conversation context

Functions:
- initialize_session_state(): Setup Streamlit session variables
- load_system(): Load or create vector database
- Main UI: Sidebar for uploads, main area for chat
"""

import streamlit as st  # Web framework for UI
import os  # File and directory operations
from pathlib import Path  # Cross-platform path handling
from src.document_processor import DocumentProcessor  # PDF processing module
from src.vector_store import VectorStore  # Vector database module
from src.rag_pipeline import RAGPipeline  # Question answering module


# ============== PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="Contract RAG Assistant",  # Browser tab title
    page_icon="📄",  # Browser tab icon
    layout="wide",  # Use full page width
    initial_sidebar_state="expanded"  # Sidebar open by default
)

# Main title
st.title("📄 Contract RAG Assistant")
st.markdown("Ask questions about your contract documents using local AI (Ollama + llama3.2)")


# ============== SESSION STATE INITIALIZATION ==============
def initialize_session_state():
    """
    Initialize Streamlit session state variables
    Session state persists data across reruns
    """
    if 'vector_store' not in st.session_state:  # Vector database instance
        st.session_state.vector_store = None
    
    if 'rag_pipeline' not in st.session_state:  # RAG pipeline instance
        st.session_state.rag_pipeline = None
    
    if 'chat_history' not in st.session_state:  # Store chat messages
        st.session_state.chat_history = []
    
    if 'system_ready' not in st.session_state:  # Is system initialized?
        st.session_state.system_ready = False

# Call initialization
initialize_session_state()


# ============== SYSTEM LOADING ==============
def load_system():
    """
    Load or create vector store and RAG pipeline
    Returns True if successful, False otherwise
    """
    try:
        # Initialize vector store
        st.session_state.vector_store = VectorStore()  # Create vector store instance
        
        # Try to load existing database
        loaded = st.session_state.vector_store.load_vectorstore()  # Load from disk
        
        if loaded:  # Database exists
            # Initialize RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(
                vector_store=st.session_state.vector_store,  # Pass vector store
                model_name="llama3.2"  # Ollama model
            )
            st.session_state.system_ready = True  # System ready to use
            return True
        else:  # No database yet
            st.warning("No database found. Please upload PDFs to get started.")
            return False
            
    except Exception as e:  # Handle errors
        st.error(f"Error loading system: {e}")
        return False


# ============== SIDEBAR - PDF UPLOAD & SETTINGS ==============
with st.sidebar:
    st.header("📁 Document Management")
    
    # PDF Upload section
    uploaded_files = st.file_uploader(
        "Upload Contract PDFs",  # Label
        type=['pdf'],  # Only accept PDFs
        accept_multiple_files=True,  # Allow multiple files
        help="Upload PDF contracts to add to the database"
    )
    
    # Process uploaded PDFs
    if uploaded_files:
        if st.button("Process Uploaded PDFs"):  # Process button
            with st.spinner("Processing PDFs..."):  # Show loading spinner
                try:
                    # Save uploaded files to disk
                    contracts_dir = Path("./data/contracts")  # Target directory
                    contracts_dir.mkdir(parents=True, exist_ok=True)  # Create if not exists
                    
                    for uploaded_file in uploaded_files:  # Loop through each file
                        file_path = contracts_dir / uploaded_file.name  # Full path
                        with open(file_path, "wb") as f:  # Write binary mode
                            f.write(uploaded_file.getbuffer())  # Save file
                    
                    st.success(f"Saved {len(uploaded_files)} PDF(s)")
                    
                    # Process PDFs into chunks
                    processor = DocumentProcessor()  # Create processor
                    chunks = processor.process_directory(str(contracts_dir))  # Process all PDFs
                    
                    # Create vector database
                    if st.session_state.vector_store is None:  # No vector store yet
                        st.session_state.vector_store = VectorStore()  # Create new one
                    
                    st.session_state.vector_store.create_vectorstore(chunks)  # Save to DB
                    
                    # Initialize RAG pipeline
                    st.session_state.rag_pipeline = RAGPipeline(
                        vector_store=st.session_state.vector_store,
                        model_name="llama3.2"
                    )
                    st.session_state.system_ready = True
                    
                    st.success("✅ PDFs processed and added to database!")
                    st.rerun()  # Reload page to show new state
                    
                except Exception as e:  # Handle errors
                    st.error(f"Error processing PDFs: {e}")
    
    st.divider()  # Visual separator
    
    # Database statistics
    st.header("📊 Database Info")
    if st.session_state.vector_store:  # If DB exists
        chunk_count = st.session_state.vector_store.get_collection_count()  # Get count
        st.metric("Total Chunks", chunk_count)  # Display metric
    else:
        st.info("No database loaded")
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    num_sources = st.slider(
        "Number of source chunks",  # Label
        min_value=1,  # Minimum
        max_value=10,  # Maximum
        value=4,  # Default
        help="How many relevant chunks to retrieve (k parameter)"
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []  # Reset chat
        if st.session_state.rag_pipeline:
            st.session_state.rag_pipeline.clear_history()  # Clear RAG history
        st.success("Chat history cleared!")
        st.rerun()  # Reload page


# ============== MAIN AREA - CHAT INTERFACE ==============

# Load system on first run
if not st.session_state.system_ready:
    load_system()

# Display chat messages
st.subheader("💬 Chat")

for message in st.session_state.chat_history:  # Loop through chat history
    with st.chat_message(message["role"]):  # Create chat bubble
        st.markdown(message["content"])  # Display message
        
        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):  # Collapsible sources
                for i, source in enumerate(message["sources"], 1):
                    st.text(f"Source {i}:")
                    st.text(source.page_content[:300] + "...")  # First 300 chars
                    if hasattr(source, 'metadata'):
                        st.caption(f"Metadata: {source.metadata}")

# Chat input
if prompt := st.chat_input("Ask a question about your contracts..."):  # User input
    
    if not st.session_state.system_ready:  # Check if system ready
        st.error("⚠️ Please upload and process PDFs first!")
    else:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):  # Show thinking indicator
                try:
                    # Query RAG pipeline
                    result = st.session_state.rag_pipeline.query(
                        question=prompt,
                        k=num_sources  # Use slider value
                    )
                    
                    answer = result['answer']  # Extract answer
                    sources = result.get('sources', [])  # Extract sources
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.text(f"Source {i}:")
                                st.text(source.page_content[:300] + "...")
                                if hasattr(source, 'metadata'):
                                    st.caption(f"Metadata: {source.metadata}")
                    
                    # Add assistant message to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:  # Handle errors
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


# ============== FOOTER ==============
st.divider()
st.caption("🤖 Powered by Ollama (llama3.2) + ChromaDB + HuggingFace Embeddings | 100% Local & Private")