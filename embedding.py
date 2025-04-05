import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import logging

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    settings,
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
from sqlalchemy import create_engine
from dotenv import load_dotenv
import config
from llama_index.core.indices.vector_store import VectorStoreIndex

load_dotenv()

logger = logging.getLogger(__name__)

class InMemoryEmbeddingStore:
    """Manages document embeddings using in-memory vector store."""
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
    ):
        """Initialize the embedding store.
        
        Args:
            embedding_model: OpenAI embedding model name
        """
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model=self.embedding_model,
            api_key=config.OPENAI_API_KEY
        )
        
        # Initialize vector store index
        self.index = VectorStoreIndex([])
        
    def process_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Process documents and store their embeddings.
        
        Args:
            documents: List of Document objects to process
            metadata: Additional metadata for the documents
            
        Returns:
            List of document IDs
        """
        try:
            logger.info(f"Processing {len(documents)} documents")
            
            # Add metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # Insert documents into the index
            for doc in documents:
                self.index.insert(doc)
            
            logger.info(f"Successfully processed {len(documents)} documents")
            return [str(i) for i in range(len(documents))]  # Return sequential IDs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
            
    def query_similar(
        self,
        query_text: str,
        num_results: int = 5,
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Query similar documents using vector similarity.
        
        Args:
            query_text: Query text to search for
            num_results: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(similarity_top_k=num_results)
            
            # Query similar documents
            results = query_engine.query(query_text)
            
            # Format results
            formatted_results = []
            for node in results.source_nodes:
                formatted_results.append({
                    "text": node.text,
                    "score": float(node.score) if node.score else 0.0,
                    "metadata": node.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying similar documents: {str(e)}")
            raise
