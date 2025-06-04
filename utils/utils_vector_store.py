import os
import re
import json
import hashlib
import shutil
import tempfile
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Import these at top level to avoid unbound references
try:
    import PyPDF2
except ImportError:
    pass

try:
    import docx
except ImportError:
    pass

# Attempt to load needed libraries - provide helpful error if missing
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.schema.document import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    langchain_available = True
except ImportError:
    langchain_available = False

try:
    import PyPDF2
    pdf_available = True
except ImportError:
    pdf_available = False

try:
    import docx
    docx_available = True
except ImportError:
    docx_available = False

# Define constants
INDEX_DIR = "vector_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class SimpleVectorStore:
    """
    A simple vector store implementation that doesn't require external libraries.
    Used as a fallback when LangChain is not available.
    """
    def __init__(self, directory: str = "simple_vector_store"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.content_file = os.path.join(directory, "content.json")
        self.index = self._load_index()
        
    def _load_index(self) -> Dict[str, Any]:
        """Load the content index from disk."""
        if os.path.exists(self.content_file):
            with open(self.content_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self) -> None:
        """Save the content index to disk."""
        with open(self.content_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def add_document(self, document_path: str, content: str) -> None:
        """Add document content to the index."""
        doc_id = hashlib.md5(document_path.encode()).hexdigest()
        chunks = self._split_text(content)
        
        # Store each chunk with its document info
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            self.index[chunk_id] = {
                "doc_path": document_path,
                "doc_name": os.path.basename(document_path),
                "content": chunk,
                "position": i
            }
        
        self._save_index()
    
    def _split_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
            
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentences to avoid breaking in the middle of sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                
                # If a single sentence is longer than chunk_size, split it
                while len(current_chunk) > chunk_size:
                    chunks.append(current_chunk[:chunk_size].strip())
                    current_chunk = current_chunk[chunk_size-overlap:].strip()
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Simple search functionality that looks for keyword matches.
        Returns the top k chunks that contain the most terms from the query.
        """
        if not self.index:
            return []
            
        # Tokenize query into terms
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Score each chunk based on term matches
        results = []
        for chunk_id, chunk_data in self.index.items():
            content = chunk_data["content"].lower()
            match_count = sum(1 for term in query_terms if term in content)
            if match_count > 0:
                results.append((chunk_data["content"], match_count))
        
        # Sort by match count (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k chunks
        return [result[0] for result in results[:top_k]]
    
    def remove_document(self, document_path: str) -> None:
        """Remove a document and its chunks from the index."""
        doc_id = hashlib.md5(document_path.encode()).hexdigest()
        new_index = {}
        
        # Keep only chunks that don't belong to the removed document
        for chunk_id, chunk_data in self.index.items():
            if not chunk_id.startswith(doc_id):
                new_index[chunk_id] = chunk_data
        
        self.index = new_index
        self._save_index()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    if not pdf_available:
        raise ImportError("PyPDF2 is required to process PDF files. Install it with 'pip install PyPDF2'.")
        
    text = ""
    try:
        # Handle the PyPDF2 import safely
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    if not docx_available:
        raise ImportError("python-docx is required to process DOCX files. Install it with 'pip install python-docx'.")
        
    text = ""
    try:
        # Handle the docx import safely
        import docx
        
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a TXT file."""
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading TXT file with latin-1 encoding: {e}")
    except Exception as e:
        print(f"Error reading TXT file: {e}")
    return text

def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return ""

def update_index_from_file(file_path: str) -> bool:
    """
    Process a file and update the vector store index.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract text from the file
        text = extract_text_from_file(file_path)
        
        if not text:
            print(f"No text could be extracted from {file_path}")
            return False
        
        # Create vector store instance
        vector_store = SimpleVectorStore()
        
        # Add document to the vector store
        vector_store.add_document(file_path, text)
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def search_index(query: str, top_k: int = 3, specific_docs: Optional[List[str]] = None) -> List[str]:
    """
    Search the vector store index for relevant document chunks.
    
    Args:
        query: Query string to search for
        top_k: Number of results to return
        specific_docs: Optional list of specific document paths to search within
        
    Returns:
        List of relevant document chunks
    """
    try:
        # Create vector store instance
        vector_store = SimpleVectorStore()
        
        # Initialize a filtered index if specific docs are provided
        if specific_docs and len(specific_docs) > 0:
            # Filter to only include chunks from the specified documents
            filtered_index = {}
            for chunk_id, chunk_data in vector_store.index.items():
                if chunk_data["doc_path"] in specific_docs:
                    filtered_index[chunk_id] = chunk_data
            
            # If we have a filtered index
            if filtered_index:
                # Create a temporary copy of the full index
                original_index = vector_store.index
                
                # Replace with filtered index temporarily
                vector_store.index = filtered_index
                
                # Search the filtered index
                results = vector_store.search(query, top_k=top_k)
                
                # Restore original index
                vector_store.index = original_index
                
                return results
        
        # If no specific docs or empty filtered index, search all
        results = vector_store.search(query, top_k=top_k)
        
        return results
        
    except Exception as e:
        print(f"Error searching index: {e}")
        return []

def remove_from_index(file_path: str) -> bool:
    """
    Remove a file from the vector store index.
    
    Args:
        file_path: Path to the file to remove
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create vector store instance
        vector_store = SimpleVectorStore()
        
        # Remove document from the vector store
        vector_store.remove_document(file_path)
        
        return True
        
    except Exception as e:
        print(f"Error removing file from index: {e}")
        return False
