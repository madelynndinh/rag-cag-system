"""
Test script for running the document retrieval and reranking functionality.
"""
import os
import logging
from typing import List
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from retrieval import (
    rerank_retrieved_documents_by_strategy,
    ReRankersInfo
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_document_retrieval():
    # Initialize OpenAI LLM
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Test query
    query = "What are the key features of Python programming language?"

    # Sample documents with different content
    documents: List[Document] = [
        Document(
            text="Python is a programming language. It was created by Guido van Rossum in 1991.",
            metadata={"file_name": "doc1.txt", "page_label": "1"}
        ),
        Document(
            text="Python is a high-level, interpreted programming language known for its simplicity and readability. "
                 "It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
            metadata={"file_name": "doc2.txt", "page_label": "1"}
        ),
        Document(
            text="Python has a comprehensive standard library that provides many built-in modules and functions. "
                 "This library is one of Python's greatest strengths, as it allows developers to accomplish many tasks "
                 "without having to write code from scratch.",
            metadata={"file_name": "doc3.txt", "page_label": "1"}
        ),
        Document(
            text="Python is widely used in web development, data science, artificial intelligence, and automation. "
                 "Frameworks like Django and Flask make it easy to build web applications.",
            metadata={"file_name": "doc4.txt", "page_label": "1"}
        ),
        Document(
            text="Python's dynamic typing and automatic memory management make it easier to write and maintain code. "
                 "The language's syntax is designed to be readable and uncluttered, with notable use of significant whitespace.",
            metadata={"file_name": "doc5.txt", "page_label": "1"}
        ),
        Document(
            text="Python has a large ecosystem of third-party packages available through the Python Package Index (PyPI). "
                 "This makes it easy to find and install libraries for almost any task.",
            metadata={"file_name": "doc6.txt", "page_label": "1"}
        ),
        Document(
            text="Python is good for beginners. It has many libraries. You can use it for web development.",
            metadata={"file_name": "doc7.txt", "page_label": "1"}
        ),
        Document(
            text="Python is a versatile programming language with several key features: "
                 "- Simple and readable syntax "
                 "- Dynamic typing and automatic memory management "
                 "- Comprehensive standard library "
                 "- Support for object-oriented, functional, and procedural programming "
                 "- Cross-platform compatibility "
                 "- Strong community support and extensive third-party packages "
                 "- Excellent for rapid prototyping and development",
            metadata={"file_name": "doc8.txt", "page_label": "1"}
        )
    ]

    # Test BM25 reranking
    print("\n" + "="*50)
    print("TESTING BM25 DOCUMENT RERANKING")
    print("="*50)
    
    print("\nCalling rerank_retrieved_documents_by_strategy with strategy='bm25'...")
    reranked_docs = rerank_retrieved_documents_by_strategy(
        query=query,
        retrieved_documents=documents,
        strategy="bm25"
    )
    
    print(f"\nReranking complete. Received {len(reranked_docs)} reranked documents")

    # Print results
    print("\nRERANKED DOCUMENTS (in order of relevance):")
    print("-"*50)
    for i, doc in enumerate(reranked_docs, 1):
        print(f"\n{i}. Document: {doc.metadata.get('file_name', 'unknown')}")
        print(f"Content: {doc.text[:200]}...")
    print("-"*50)

    # Test QueryFusionRetriever reranking
    print("\n" + "="*50)
    print("TESTING QUERYFUSION DOCUMENT RERANKING")
    print("="*50)
    
    print("\nCalling rerank_retrieved_documents_by_strategy with strategy='queryfusion'...")
    reranked_docs = rerank_retrieved_documents_by_strategy(
        query=query,
        retrieved_documents=documents,
        strategy="queryfusion"
    )
    
    print(f"\nReranking complete. Received {len(reranked_docs)} reranked documents")

    # Print results
    print("\nRERANKED DOCUMENTS (in order of relevance):")
    print("-"*50)
    for i, doc in enumerate(reranked_docs, 1):
        print(f"\n{i}. Document: {doc.metadata.get('file_name', 'unknown')}")
        print(f"Content: {doc.text[:200]}...")
    print("-"*50)

def main():
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        print("For testing purposes, we'll proceed with mock data.")
        print("To use real OpenAI, run: export OPENAI_API_KEY=your_key_here\n")
    
    try:
        # Test document retrieval and reranking
        test_document_retrieval()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Please make sure you have valid credentials and proper network connection.")

if __name__ == "__main__":
    main()
