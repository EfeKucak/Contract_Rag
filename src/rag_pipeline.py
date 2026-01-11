"""
RAG Pipeline Module
Connects vector store with Ollama LLM for question answering

Main Functions Overview:
1. __init__(): Initialize pipeline with vector store and LLM model
2. create_prompt(): Build structured prompt with context for LLM
3. query(): Main RAG function - Retrieve docs, create prompt, get answer from Ollama
4. query_with_history(): Same as query() but remembers conversation history
5. clear_history(): Reset conversation history for fresh start
6. format_sources(): Display source documents in readable format
"""

from typing import List, Dict  # Type hints
import ollama  # Local LLM integration
from .vector_store import VectorStore  # Our vector store module


class RAGPipeline:
    """RAG pipeline for contract question answering"""
    
    def __init__(self, vector_store: VectorStore, model_name: str = "llama3.2"):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Initialized vector store instance
            model_name: Ollama model name to use
        """
        self.vector_store = vector_store  # Vector database for document retrieval
        self.model_name = model_name  # LLM model name (llama3.2)
        self.chat_history = []  # Store conversation history


    def create_prompt(self, query: str, context_docs: List) -> str:
        """
        Create prompt with context for LLM
        
        Args:
            query: User's question
            context_docs: Relevant document chunks from vector store
            
        Returns:
            Formatted prompt string
        """
        # Extract text content from documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create structured prompt
        prompt = f"""You are a helpful assistant that answers questions about contracts and agreements.
        Use the following context from the contract documents to answer the question.
        If you cannot find the answer in the context, say "I cannot find this information in the provided contracts."

        Context from contracts:
        {context}
        

        Question: {query}

        Answer:"""
        
        return prompt  # Return formatted prompt
    
    
    def query(self, question: str, k: int = 4) -> Dict[str, any]:
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        # Step 1: Retrieve relevant documents from vector store
        print(f"Searching for relevant documents...")
        relevant_docs = self.vector_store.search(question, k=k)  # Find k most similar chunks
        
        if not relevant_docs:  # No documents found
            return {
                "answer": "No relevant documents found in the database.",
                "sources": []
            }
        
        # Step 2: Create prompt with context
        prompt = self.create_prompt(question, relevant_docs)  # Build LLM prompt
        
        # Step 3: Get answer from Ollama
        print(f"Asking {self.model_name}...")
        response = ollama.chat(
            model=self.model_name,  # Use llama3.2
            messages=[{"role": "user", "content": prompt}]  # Send prompt to LLM
        )
        
        # Step 4: Extract answer
        answer = response['message']['content']  # Get LLM's response text
        
        # Step 5: Return answer with sources
        return {
            "answer": answer,  # LLM's answer
            "sources": relevant_docs,  # Documents used for context
            "num_sources": len(relevant_docs)  # How many chunks were used
        }
    
    def query_with_history(self, question: str, k: int = 4) -> Dict[str, any]:
        """
        Answer question with conversation history support
        
        Args:
            question: User's question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        # Get relevant documents
        relevant_docs = self.vector_store.search(question, k=k)
        
        if not relevant_docs:
            return {
                "answer": "No relevant documents found in the database.",
                "sources": []
            }
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Build messages with history
        messages = self.chat_history.copy()  # Copy existing conversation
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })
        
        # Get response from Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=messages  # Send full conversation history
        )
        
        answer = response['message']['content']
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return {
            "answer": answer,
            "sources": relevant_docs,
            "num_sources": len(relevant_docs)
        }
        
    def clear_history(self) -> None:
        """
        Clear conversation history
        """
        self.chat_history = []  # Reset chat history
        print("Chat history cleared.")
        
        
    def format_sources(self, sources: List) -> str:
        """
        Format source documents for display
        
        Args:
            sources: List of source documents
            
        Returns:
            Formatted string of sources
        """
        if not sources:  # No sources
            return "No sources available."
        
        formatted = "\n\n" + "="*50 + "\nSOURCES:\n" + "="*50 + "\n"
        
        for i, doc in enumerate(sources, 1):  # Loop through each source
            formatted += f"\n[Source {i}]\n"
            formatted += f"Content: {doc.page_content[:200]}...\n"  # First 200 chars
            
            # Add metadata if available
            if hasattr(doc, 'metadata') and doc.metadata:
                formatted += f"Metadata: {doc.metadata}\n"
        
        return formatted  # Return formatted sources
        
    