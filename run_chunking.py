from typing import List, Dict, Any
from llama_index.core import Document
from draft.chunking import ChunkingService, ChunkingInfo
from llama_index.readers.file import PDFReader
import logging
import os
from pathlib import Path
import random
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf_documents(folders: List[str], max_files: int = 1) -> List[Document]:
    """Load PDF documents from specified folders."""
    documents = []
    pdf_reader = PDFReader()
    
    for folder in folders:
        folder_path = Path("./pdf-test")
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder}")
            continue
            
        logger.info(f"Processing PDFs in folder: {folder}")
        pdf_files = list(folder_path.glob("**/*.pdf"))
        
        # Randomly select up to max_files PDFs
        selected_files = random.sample(pdf_files, min(max_files, len(pdf_files)))
        
        for pdf_path in selected_files:
            try:
                logger.info(f"Loading PDF: {pdf_path}")
                # Load PDF and create documents
                pdf_docs = pdf_reader.load_data(str(pdf_path))
                
                # Add source metadata to each document
                for doc in pdf_docs:
                    doc.metadata.update({
                        "source": str(pdf_path),
                        "file_name": pdf_path.name,
                        "folder": folder
                    })
                
                documents.extend(pdf_docs)
                logger.info(f"Successfully loaded {len(pdf_docs)} documents from {pdf_path}")
                
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_path}: {str(e)}", exc_info=True)
    
    return documents

def analyze_document_hierarchy(original_docs: List[Document], parent_docs: List[Document], chunked_docs: List[Any]) -> None:
    """Analyze and display the document hierarchy and relationships."""
    logger.info("\n=== Document Hierarchy Analysis ===")
    
    # 1. Original Documents Analysis
    logger.info("\n1. Original Documents:")
    for i, doc in enumerate(original_docs):
        logger.info(f"\nOriginal Document {i+1}:")
        logger.info(f"Source: {doc.metadata.get('source', 'unknown')}")
        logger.info(f"File Name: {doc.metadata.get('file_name', 'unknown')}")
        logger.info(f"Content Length: {len(doc.text.split())} words")
        logger.info(f"Metadata: {doc.metadata}")
        
    # 2. Parent Documents Analysis
    logger.info("\n2. Parent Documents:")
    for i, parent in enumerate(parent_docs):
        logger.info(f"\nParent Document {i+1}:")
        logger.info(f"Parent UUID: {parent.metadata.get('parent_uuid', 'unknown')}")
        logger.info(f"Source: {parent.metadata.get('source', 'unknown')}")
        logger.info(f"Content Length: {len(parent.text.split())} words")
        logger.info(f"Metadata: {parent.metadata}")

    # 3. Chunks Analysis
    # Group chunks by parent UUID
    chunks_by_parent = defaultdict(list)
    for chunk in chunked_docs:
        if isinstance(chunk, Document):
            parent_uuid = chunk.metadata.get('parent_uuid', 'unknown')
            chunks_by_parent[parent_uuid].append(chunk)
        else:
            chunks_by_parent['unknown'].append(chunk)

    logger.info("\n3. Parent-Child Relationships:")
    for parent_uuid, chunks in chunks_by_parent.items():
        logger.info(f"\nParent UUID: {parent_uuid}")
        logger.info(f"Number of child chunks: {len(chunks)}")
        
        # Display sample chunks for this parent
        logger.info("Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks per parent
            logger.info(f"\n  Chunk {i+1}:")
            if isinstance(chunk, Document):
                logger.info(f"  Content: {chunk.text[:]}...")
                logger.info(f"  Size: {len(chunk.text.split())} words")
                logger.info(f"  Metadata: {chunk.metadata}")
            else:
                logger.info(f"  Content: {str(chunk)[:]}...")
                logger.info(f"  Size: {len(str(chunk).split())} words")

def main():
    """Main function to test chunking functionality with PDF documents."""
    try:
        # Initialize chunking service
        chunking_service = ChunkingService()
        
        # Define folders to process
        folders = [
            "pdf-test"
        ]
        
        # Load PDF documents (limit to 10 files)
        documents = load_pdf_documents(folders, max_files=10)
        if not documents:
            logger.error("No documents were loaded. Please check the folders and PDF files.")
            return
            
        logger.info(f"Loaded {len(documents)} documents from PDFs")
        
        # Test website document chunking
        logger.info("\n=== Testing Document Chunking ===")
        parent_docs, chunked_docs = chunking_service.chunk_website_documents(
            documents=documents,
            APPROX_ALLOWED_TOKENS_PER_PARENT_DOCUMENT=2000
        )
        
        # Analyze document hierarchy
        analyze_document_hierarchy(documents, parent_docs, chunked_docs)
        
        # Test parent document generation for a single document
        logger.info("\n=== Testing Single Document Parent Generation ===")
        single_parent_docs, nodes = chunking_service.generate_parent_document_for_given_documents(
            documents=[documents[0]],  # Test with first document
            APPROX_ALLOWED_TOKENS_PER_PARENT_DOCUMENT=2000
        )
        
        logger.info("\n4. Single Document Parent-Child Analysis:")
        logger.info(f"Original document source: {documents[0].metadata.get('source', 'unknown')}")
        logger.info(f"Number of parent documents: {len(single_parent_docs)}")
        logger.info(f"Number of nodes: {len(nodes)}")
        
        # Display node hierarchy
        logger.info("\nNode Hierarchy:")
        for i, node in enumerate(nodes):
            indent = "  " * node.metadata.get("level", 0)  # Indent based on node level
            logger.info(f"\n{indent}Node {i+1}:")
            logger.info(f"{indent}Level: {node.metadata.get('level', 'unknown')}")
            logger.info(f"{indent}Content: {node.text[:]}")
            logger.info(f"{indent}Metadata: {node.metadata}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 