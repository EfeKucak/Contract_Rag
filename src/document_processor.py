"""
Document Processor Module
Handles PDF loading, text extraction, and chunking for RAG
"""

from typing import List  # Type hints for clarity
from pathlib import Path  # Handle file paths across different operating systems
from langchain_community.document_loaders import PyPDFLoader  # Load and read PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text into smaller chunks


class DocumentProcessor:
    """Process PDF documents for RAG pipeline"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks for context
        """
        self.chunk_size = chunk_size  # Each text part will be ~1000 characters
        self.chunk_overlap = chunk_overlap  # 200 char overlap for context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> List:
        """
        Load and extract text from PDF

        Args:
            file_path: Path to PDF file

        Returns:
            List of document chunks
        """
        loader = PyPDFLoader(file_path)  # Create PDF loader
        documents = loader.load()  # Load all pages
        return documents  # Return list of documents

    def chunk_documents(self, documents: List) -> List:
        """
        Split documents into chunks

        Args:
            documents: List of documents

        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)  # Split into chunks
        return chunks  # Return chunked documents

    def process_pdf(self, file_path: str) -> List:
        """
        Complete pipeline: load PDF and chunk it

        Args:
            file_path: Path to PDF file

        Returns:
            List of processed chunks ready for vectorization
        """
        print(f"Loading PDF: {file_path}")
        documents = self.load_pdf(file_path)  # Load PDF

        print(f"Loaded {len(documents)} pages")
        chunks = self.chunk_documents(documents)  # Chunk the documents

        print(f"Created {len(chunks)} chunks")
        return chunks  # Return all chunks

    def process_directory(self, directory_path: str) -> List:
        """
        Process all PDF files in a directory

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of all processed chunks from all PDFs
        """
        all_chunks = []  # Collect all chunks
        pdf_files = Path(directory_path).glob("*.pdf")  # Find all PDFs

        for pdf_file in pdf_files:  # Loop through each PDF
            print(f"\nProcessing: {pdf_file.name}")
            chunks = self.process_pdf(str(pdf_file))  # Process this PDF
            all_chunks.extend(chunks)  # Add to collection

        print(f"\nTotal chunks from all documents: {len(all_chunks)}")
        return all_chunks  # Return all chunks from all PDFs
