from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
import config
import os
import tempfile
import re
from typing import List, Optional
import unicodedata
import logging

logger = logging.getLogger(__name__)

async def create_index_from_file(file_path: Path):
    """Create a vector index from a file"""
    try:
        config.logger.info(f"Creating index for file: {file_path}")
        
        # Load documents
        docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        
        # Create index
        index = VectorStoreIndex.from_documents(docs)
        
        # Get filename without extension
        file_name = file_path.stem
        
        # Save index
        storage_path = Path(config.STORAGE_DIR) / file_name
        index.storage_context.persist(persist_dir=str(storage_path))
        
        config.logger.info(f"Successfully created and stored index for: {file_path}")
        return str(storage_path)
    except Exception as e:
        config.logger.error(f"Error creating index for file {file_path}: {str(e)}")
        raise e

async def query_document(query: str, index_path: str, similarity_top_k: int = 3):
    """Query a document index"""
    try:
        config.logger.info(f"Querying document with index at: {index_path}")
        
        # Load index
        index = VectorStoreIndex.load_from_disk(index_path)
        
        # Create retriever with similarity top k
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k
        )
        
        # Create query engine with similarity cutoff
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )
        
        # Query
        response = query_engine.query(query)
        
        config.logger.info(f"Successfully queried document")
        return response
    except Exception as e:
        config.logger.error(f"Error querying document: {str(e)}")
        raise e

async def create_temp_file_from_content(content: bytes, file_name: str):
    """Create a temporary file from content"""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file_name)
        
        # Write content to file
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        config.logger.info(f"Created temporary file: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        config.logger.error(f"Error creating temporary file: {str(e)}")
        raise e

def preprocess_pdf(text: str) -> str:
    """Clean and normalize text extracted from PDF.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
        
    try:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction artifacts
        text = fix_pdf_artifacts(text)
        
        # Normalize quotes and dashes
        text = normalize_punctuation(text)
        
        # Clean up final text
        text = text.strip()
        
        return text
        
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return text

def fix_pdf_artifacts(text: str) -> str:
    """Fix common artifacts from PDF text extraction.
    
    Args:
        text: Text with potential PDF artifacts
        
    Returns:
        Cleaned text
    """
    # Remove form feed characters
    text = text.replace('\f', '\n')
    
    # Fix broken words (words split by newlines)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Remove header/footer artifacts (e.g., page numbers)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Fix bullets and numbered lists
    text = re.sub(r'•\s*', '- ', text)
    text = re.sub(r'^\d+\.\s+', '\\0', text, flags=re.MULTILINE)  # Keep the original number
    
    return text

def normalize_punctuation(text: str) -> str:
    """Normalize various types of quotes, dashes, and other punctuation.
    
    Args:
        text: Text with potentially inconsistent punctuation
        
    Returns:
        Text with normalized punctuation
    """
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    
    # Normalize ellipsis
    text = text.replace('…', '...')
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    
    return text

def clean_text_for_embedding(text: str) -> str:
    """Prepare text for embedding by removing unnecessary elements.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text suitable for embedding
    """
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    
    # Remove special characters and numbers (keep basic punctuation)
    text = re.sub(r'[^a-zA-Z\s.,!?;:-]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()
