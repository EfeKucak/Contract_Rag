"""
Vector Store Module
Manages ChromaDB for storing and retrieving document embeddings
"""

from typing import List, Optional  # Type hints for clarity
import chromadb  # Vector database for storing embeddings
from chromadb.config import Settings  # ChromaDB configuration
from langchain_community.embeddings import HuggingFaceEmbeddings  # Convert text to vectors
from langchain_community.vectorstores import Chroma  # LangChain's ChromaDB wrapper


class VectorStore:
    """Manage vector database for document retrieval"""

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize vector store

        Args:
            persist_directory: Directory to save the database
        """
        self.persist_directory = persist_directory  # Where to save the database
        self.embeddings = HuggingFaceEmbeddings(  # Multilingual embedding model for text-to-vector conversion
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = None  # Will be initialized later (load or create)

    def create_vectorstore(self, documents: List) -> None:
        """
        Create vector store from documents

        Args:
            documents: List of document chunks
        """
        print(f"Creating embeddings for {len(documents)} chunks...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,  # chunks which come from document_processor
            embedding=self.embeddings,  # Hugging Face Model
            persist_directory=self.persist_directory  # Save DB to the disk
        )
        print("Vector store created and saved!")

    def load_vectorstore(self) -> bool:
        """
        Load existing vector store from disk

        Returns:
            True if successful, False otherwise
        """
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,  # Load from disk
                embedding_function=self.embeddings  # Use same embedding model
            )
            print("Vector store loaded successfully!")
            return True  # Successfully loaded
        except Exception as e:
            print(f"Could not load vector store: {e}")
            return False  # Failed to load (maybe DB doesn't exist yet)

    def search(self, query: str, k: int = 4) -> List:
        """
        Search for relevant documents

        Args:
            query: User's question
            k: Number of most relevant chunks to return

        Returns:
            List of relevant document chunks
        """
        if self.vectorstore is None:  # Check if DB is loaded
            print("Vector store not initialized!")
            return []  # Return empty list

        results = self.vectorstore.similarity_search(query, k=k)  # Find k most similar chunks
        return results  # Return relevant document chunks

    def get_collection_count(self) -> int:
        """
        Get number of documents in vector store

        Returns:
            Number of chunks stored in database
        """
        if self.vectorstore is None:  # Check if DB exists
            return 0  # No chunks if DB not loaded

        collection = self.vectorstore._collection  # Access ChromaDB collection
        return collection.count()  # Return total number of chunks stored
