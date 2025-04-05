"""
Test script for running the reranker functionality.
"""
import os
from typing import List, Tuple
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from rerankers import (
    rerank_llm_responses_by_strategy,
    rerank_retrieved_documents_by_strategy,
    ReRankersInfo
)

def test_llm_rerank():
    # Initialize OpenAI LLM
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Test query
    query = "What are the key features of Python programming language?"

    # Sample responses with different quality levels
    responses: List[Tuple[str, str]] = [
        ("resp1", "Python is a programming language. It was created by Guido van Rossum."),
        ("resp2", """Python is a high-level, interpreted programming language known for its simplicity and readability. 
        Key features include:
        1. Dynamic typing
        2. Automatic memory management
        3. Extensive standard library
        4. Support for multiple programming paradigms
        5. Large ecosystem of third-party packages
        Python is widely used in web development, data science, AI, and automation."""),
        ("resp3", "Python is good for beginners. It has many libraries. You can use it for web development."),
        ("resp4", """Python is a versatile programming language with several key features:
        - Simple and readable syntax
        - Dynamic typing and automatic memory management
        - Comprehensive standard library
        - Support for object-oriented, functional, and procedural programming
        - Cross-platform compatibility
        - Strong community support and extensive third-party packages
        - Excellent for rapid prototyping and development""")
    ]

    # Test LLM reranking
    print("\n" + "="*50)
    print("TESTING LLM RERANKING WITH STRATEGY 'llm_rerank'")
    print("="*50)
    
    print("\nCalling rerank_llm_responses_by_strategy with strategy='llm_rerank'...")
    reranked_ids = rerank_llm_responses_by_strategy(
        query=query,
        responses=responses,
        strategy="llm_rerank",
        llm=llm
    )
    
    print(f"\nReranking complete. Received {len(reranked_ids)} reranked IDs: {reranked_ids}")

    # Print results
    print("\nRERANKED RESPONSES (in order of relevance):")
    print("-"*50)
    for i, doc_id in enumerate(reranked_ids, 1):
        response = next(resp for resp_id, resp in responses if resp_id == doc_id)
        print(f"\n{i}. Response ID: {doc_id}")
        print(f"Content: {response[:200]}...")
    print("-"*50)

def test_sentence_transformer_rerank():
    print("\n" + "="*50)
    print("TESTING SENTENCE TRANSFORMER RERANKING")
    print("="*50)
    
    # Test query
    query = "What are the key features of Python programming language?"

    # Sample responses with different quality levels
    responses: List[Tuple[str, str]] = [
        ("resp1", "Python is a programming language. It was created by Guido van Rossum."),
        ("resp2", """Python is a high-level, interpreted programming language known for its simplicity and readability. 
        Key features include:
        1. Dynamic typing
        2. Automatic memory management
        3. Extensive standard library
        4. Support for multiple programming paradigms
        5. Large ecosystem of third-party packages
        Python is widely used in web development, data science, AI, and automation."""),
        ("resp3", "Python is good for beginners. It has many libraries. You can use it for web development."),
        ("resp4", """Python is a versatile programming language with several key features:
        - Simple and readable syntax
        - Dynamic typing and automatic memory management
        - Comprehensive standard library
        - Support for object-oriented, functional, and procedural programming
        - Cross-platform compatibility
        - Strong community support and extensive third-party packages
        - Excellent for rapid prototyping and development""")
    ]

    print("\nCalling rerank_llm_responses_by_strategy with strategy='sentence_transformer'...")
    reranked_ids = rerank_llm_responses_by_strategy(
        query=query,
        responses=responses,
        strategy="sentence_transformer"
    )
    
    print(f"\nReranking complete. Received {len(reranked_ids)} reranked IDs: {reranked_ids}")

    print("\nRERANKED RESPONSES (in order of relevance):")
    print("-"*50)
    for i, doc_id in enumerate(reranked_ids, 1):
        response = next(resp for resp_id, resp in responses if resp_id == doc_id)
        print(f"\n{i}. Response ID: {doc_id}")
        print(f"Content: {response[:200]}...")
    print("-"*50)

def main():
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        print("For testing purposes, we'll proceed with mock data.")
        print("To use real OpenAI, run: export OPENAI_API_KEY=your_key_here\n")
    
    # Initialize LLM for response reranking
    try:
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
        
        # Test all reranking strategies
        test_llm_rerank()
        test_sentence_transformer_rerank()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Please make sure you have valid credentials and proper network connection.")

if __name__ == "__main__":
    main() 