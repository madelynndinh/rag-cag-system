"""
RAG Controller - A comprehensive implementation of Retrieval-Augmented Generation.

This module integrates document parsing, embeddings, retrieval, and reranking
to provide a complete RAG system for answering questions based on document content.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore

# Import our custom modules
from parsing import parse_pdf
from embedding import InMemoryEmbeddingStore
from retrieval import (
    rerank_retrieved_documents_by_strategy,
    ReRankersInfo
)
from rerankers import (
    rerank_llm_responses_by_strategy,
    ReRankersInfo as LLMReRankersInfo
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from the RAG system."""
    response: str
    context_used: List[Document]
    metadata: Dict[str, Any]

class RAGController:
    """
    A comprehensive RAG controller that integrates parsing, embeddings, retrieval, and reranking.
    
    This class provides a complete RAG system for answering questions based on document content.
    """
    
    def __init__(
        self,
        llm: Optional[OpenAI] = None,
        embedding_store: Optional[InMemoryEmbeddingStore] = None,
        reranking_strategy: str = ReRankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_RETRIEVED_DOCUMENTS.value,
        llm_reranking_strategy: str = LLMReRankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_LLM_RESPONSES.value,
        max_context_docs: int = 4
    ):
        """
        Initialize the RAG controller.
        
        Args:
            llm: The language model to use for response generation
            embedding_store: The embedding store to use for document retrieval
            reranking_strategy: The strategy to use for reranking documents
            llm_reranking_strategy: The strategy to use for reranking LLM responses
            max_context_docs: Maximum number of documents to use as context
        """
        self.llm = llm or OpenAI(model="gpt-4-turbo-preview", temperature=0.7)
        self.embedding_store = embedding_store or InMemoryEmbeddingStore()
        self.reranking_strategy = reranking_strategy
        self.llm_reranking_strategy = llm_reranking_strategy
        self.max_context_docs = max_context_docs
        
        # Validate reranking strategies
        if reranking_strategy not in ReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_RETRIEVED_DOCUMENTS.value:
            raise ValueError(
                f"Invalid document reranking strategy. Must be one of: "
                f"{ReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_RETRIEVED_DOCUMENTS.value}"
            )
            
        if llm_reranking_strategy not in LLMReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES.value:
            raise ValueError(
                f"Invalid LLM reranking strategy. Must be one of: "
                f"{LLMReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES.value}"
            )
    
    def process_documents(self, documents: List[Document], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process documents and store them in the embedding store.
        
        Args:
            documents: List of Document objects to process
            metadata: Optional metadata to associate with the documents
            
        Returns:
            List of document IDs
        """
        try:
            logger.info(f"Processing {len(documents)} documents")
            
            # Process documents in the embedding store
            doc_ids = self.embedding_store.process_documents(documents, metadata)
            
            logger.info(f"Successfully processed {len(doc_ids)} documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def retrieve_documents(self, query: str, num_results: int = 10) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: The query to search for
            num_results: Number of results to retrieve
            
        Returns:
            List of relevant Document objects
        """
        try:
            logger.info(f"Retrieving documents for query: {query}")
            
            # Query the embedding store
            results = self.embedding_store.query_similar(query, num_results=num_results)
            
            # Convert results to Document objects
            documents = []
            for result in results:
                doc = Document(
                    text=result["text"],
                    metadata=result["metadata"]
                )
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents based on relevance to the query.
        
        Args:
            query: The query to rerank for
            documents: List of Document objects to rerank
            
        Returns:
            List of reranked Document objects
        """
        try:
            logger.info(f"Reranking {len(documents)} documents for query: {query}")
            
            # Rerank documents using the specified strategy
            reranked_docs = rerank_retrieved_documents_by_strategy(
                query=query,
                retrieved_documents=documents,
                strategy=self.reranking_strategy
            )
            
            # Limit the number of context documents
            context_docs = reranked_docs[:self.max_context_docs]
            
            logger.info(f"Reranked documents, using {len(context_docs)} as context")
            return context_docs
            
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            raise
    
    def generate_multiple_responses(
        self,
        query: str,
        context_docs: List[Document],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        num_responses: int = 3
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate multiple responses using the LLM.
        
        Args:
            query: The user's query
            context_docs: List of context documents
            system_prompt: Optional system prompt to guide response generation
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens in the response
            num_responses: Number of different responses to generate
            
        Returns:
            List of tuples containing (response, metadata)
        """
        try:
            logger.info(f"Generating {num_responses} responses for query: {query}")
            
            responses = []
            
            # Prepare context string
            context_str = "\n\n".join([
                f"Document {i+1}:\n{doc.text}"
                for i, doc in enumerate(context_docs)
            ])
            
            # Generate multiple responses with different temperatures
            for i in range(num_responses):
                # Vary temperature slightly for diversity
                current_temp = temperature + (i * 0.1)
                
                # Prepare prompt
                prompt = ""
                if system_prompt:
                    prompt += f"{system_prompt}\n\n"
                
                prompt += f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
                
                # Generate response using the LLM
                response = self.llm.complete(
                    prompt=prompt,
                    temperature=current_temp,
                    max_tokens=max_tokens
                ).text
                
                # Create metadata
                metadata = {
                    "response_id": f"resp{i+1}",
                    "temperature": current_temp,
                    "context_docs": len(context_docs)
                }
                
                responses.append((response, metadata))
            
            logger.info(f"Generated {len(responses)} responses")
            return responses
            
        except Exception as e:
            logger.error(f"Error generating responses: {str(e)}")
            raise
    
    def rerank_responses(
        self,
        query: str,
        responses: List[Tuple[str, Dict[str, Any]]]
    ) -> List[str]:
        """
        Rerank responses based on relevance to the query.
        
        Args:
            query: The query to rerank for
            responses: List of (response, metadata) tuples to rerank
            
        Returns:
            List of reranked response IDs
        """
        try:
            logger.info(f"Reranking {len(responses)} responses for query: {query}")
            
            # Extract response texts and IDs
            response_texts = [resp[0] for resp in responses]
            response_ids = [resp[1]["response_id"] for resp in responses]
            
            # Rerank responses using the specified strategy
            reranked_ids = rerank_llm_responses_by_strategy(
                query=query,
                responses=list(zip(response_ids, response_texts)),  # Format as (id, text) tuples
                strategy=self.llm_reranking_strategy,
                llm=self.llm  # Pass the LLM instance
            )
            
            logger.info(f"Reranked responses: {reranked_ids}")
            return reranked_ids
            
        except Exception as e:
            logger.error(f"Error reranking responses: {str(e)}")
            raise
    
    def generate_response(
        self,
        query: str,
        context_docs: List[Document],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_llm_reranking: bool = True,
        num_responses: int = 3
    ) -> RAGResponse:
        """
        Generate a response using the RAG system.
        
        Args:
            query: The user's query
            context_docs: List of context documents
            system_prompt: Optional system prompt to guide response generation
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens in the response
            use_llm_reranking: Whether to use LLM reranking for responses
            num_responses: Number of different responses to generate if using LLM reranking
            
        Returns:
            RAGResponse containing the generated response and metadata
        """
        try:
            # Generate multiple responses if using LLM reranking
            if use_llm_reranking and num_responses > 1:
                # Generate multiple responses
                logger.info(f"Generating {num_responses} responses for LLM reranking")
                responses = self.generate_multiple_responses(
                    query=query,
                    context_docs=context_docs,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_responses=num_responses
                )
                
                # Extract response texts and IDs
                response_texts = [resp[0] for resp in responses]
                response_ids = [resp[1]["response_id"] for resp in responses]
                
                # Rerank responses
                logger.info(f"Reranking {len(response_texts)} responses using strategy: {self.llm_reranking_strategy}")
                reranked_ids = self.rerank_responses(query, responses)
                
                # Get the top response
                if reranked_ids:
                    top_response_id = reranked_ids[0]
                    top_response_idx = response_ids.index(top_response_id)
                    final_response = response_texts[top_response_idx]
                    response_metadata = responses[top_response_idx][1]
                    response_metadata["reranked_position"] = 1
                else:
                    # Fallback to the first response if reranking fails
                    final_response = response_texts[0]
                    response_metadata = responses[0][1]
                    response_metadata["reranked_position"] = 0
            else:
                # Generate a single response
                context_str = "\n\n".join([
                    f"Document {i+1}:\n{doc.text}"
                    for i, doc in enumerate(context_docs)
                ])
                
                # Prepare prompt
                prompt = ""
                if system_prompt:
                    prompt += f"{system_prompt}\n\n"
                
                prompt += f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
                
                final_response = self.llm.complete(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                ).text
                
                response_metadata = {
                    "response_id": "resp1",
                    "temperature": temperature,
                    "context_docs": len(context_docs)
                }
            
            # Create response object with metadata
            return RAGResponse(
                response=final_response,
                context_used=context_docs,
                metadata={
                    "reranking_strategy": self.reranking_strategy,
                    "llm_reranking_strategy": self.llm_reranking_strategy if use_llm_reranking else None,
                    "num_context_docs": len(context_docs),
                    "response_metadata": response_metadata
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def process_pdf_and_query(
        self,
        pdf_path: str,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_llm_reranking: bool = True,
        num_responses: int = 3
    ) -> RAGResponse:
        """
        Process a PDF file and answer a query about its content.
        
        Args:
            pdf_path: Path to the PDF file
            query: The query to answer
            system_prompt: Optional system prompt to guide response generation
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens in the response
            use_llm_reranking: Whether to use LLM reranking for responses
            num_responses: Number of different responses to generate if using LLM reranking
            
        Returns:
            RAGResponse containing the generated response and metadata
        """
        try:
            # Parse the PDF using imported parse_pdf function
            documents = parse_pdf(pdf_path)
            
            # Process documents and store in embedding store
            doc_ids = self.process_documents(documents)
            
            # Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(query)
            
            # Rerank documents
            context_docs = self.rerank_documents(query, retrieved_docs)
            
            # Generate response
            response = self.generate_response(
                query=query,
                context_docs=context_docs,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                use_llm_reranking=use_llm_reranking,
                num_responses=num_responses
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing PDF and query: {str(e)}")
            raise

def test_rag_controller():
    """Test the RAG controller with all PDF files in the pdf-test directory."""
    # Initialize RAG controller
    rag = RAGController(
        reranking_strategy="bm25",  # Using BM25 for precise matching
        llm_reranking_strategy="llm_rerank",  # Using LLM reranking for responses
        max_context_docs=4  # Allow for more context
    )
    
    # Get all PDF files in the pdf-test directory
    import glob
    
    pdf_dir = "pdf-test"
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir} directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Parse all PDF files first
    all_documents = []
    for pdf_path in pdf_files:
        print(f"\nParsing PDF: {pdf_path}")
        
        # Parse the PDF using imported parse_pdf function
        documents = parse_pdf(pdf_path)
        all_documents.extend(documents)
        
        print(f"Parsed {len(documents)} document blocks from {pdf_path}")
    
    print(f"\nTotal documents parsed: {len(all_documents)}")
    
    # Process all documents at once
    print("\nProcessing all documents in batch...")
    doc_ids = rag.process_documents(all_documents)
    print(f"Successfully processed {len(doc_ids)} documents")
    
    # Prompt user for query
    print("\n" + "="*50)
    print("RAG SYSTEM READY FOR QUERIES")
    print("="*50)
    
    while True:
        # Get user query
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
            
        # Get optional system prompt
        system_prompt = input("\nEnter a system prompt (optional, press Enter to skip): ")
        if not system_prompt.strip():
            system_prompt = None
            
        # Retrieve relevant documents
        print(f"\nRetrieving documents for query: {query}")
        retrieved_docs = rag.retrieve_documents(query)
        
        # Rerank documents
        print(f"Reranking {len(retrieved_docs)} documents")
        context_docs = rag.rerank_documents(query, retrieved_docs)
        
        # Generate response
        print("Generating response...")
        response = rag.generate_response(
            query=query,
            context_docs=context_docs,
            system_prompt=system_prompt,
            temperature=0.7,
            use_llm_reranking=True,
            num_responses=3
        )
        
        # Print results
        print("\n" + "="*50)
        print("RESPONSE")
        print("="*50)
        print("\nQuery:", query)
        print("\nGenerated Response:", response.response)
        print("\nContext Documents Used:")
        for i, doc in enumerate(response.context_used):
            print(f"\nDocument {i+1}:")
            print(f"Source: {doc.metadata['source']}")
            print(f"Type: {doc.metadata['type']}")
            if 'level' in doc.metadata:
                print(f"Level: {doc.metadata['level']}")
            print(f"Page: {doc.metadata['page_num']}")
            print(f"Content: {doc.text}")
        print("\nMetadata:", response.metadata)
        print("\n" + "="*50)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_rag_controller()
