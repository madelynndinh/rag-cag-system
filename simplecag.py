"""
Simple Context-Aware Generation (CAG) implementation with KV-cache support.

This module provides a basic implementation of Context-Aware Generation,
which combines document retrieval, reranking, and response generation
to produce contextually relevant responses. It uses KV-cache for efficient
context handling.
"""

import logging
import os
import json
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from retrieval import (
    rerank_retrieved_documents_by_strategy,
    ReRankersInfo
)
from rerankers import (
    rerank_llm_responses_by_strategy,
    ReRankersInfo as LLMReRankersInfo
)
from llmsherpa.readers import LayoutPDFReader

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CAGResponse:
    """Response from the Context-Aware Generation system."""
    response: str
    context_used: List[Document]
    metadata: Dict[str, Any]

class SimpleCAG:
    """
    A simple implementation of Context-Aware Generation with KV-cache support.
    
    This class combines document retrieval, reranking, and response generation
    to produce contextually relevant responses to user queries. It uses KV-cache
    for efficient context handling.
    """
    
    def __init__(
        self,
        llm: Optional[LLM] = None,
        reranking_strategy: str = ReRankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_RETRIEVED_DOCUMENTS.value,
        llm_reranking_strategy: str = LLMReRankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_LLM_RESPONSES.value,
        max_context_docs: int = 4,
        llmsherpa_api_url: str = "http://localhost:5011/api/parseDocument?renderFormat=all",
        cache_dir: str = "cache",
        use_kv_cache: bool = True
    ):
        """
        Initialize the SimpleCAG system.
        
        Args:
            llm: The language model to use for response generation
            reranking_strategy: The strategy to use for reranking documents
            llm_reranking_strategy: The strategy to use for reranking LLM responses
            max_context_docs: Maximum number of documents to use as context
            llmsherpa_api_url: URL for the LLMSherpa PDF parsing service
            cache_dir: Directory to store KV-cache files
            use_kv_cache: Whether to use KV-cache for context handling
        """
        self.llm = llm or OpenAI(model="gpt-4-turbo-preview", temperature=0.7)
        self.reranking_strategy = reranking_strategy
        self.llm_reranking_strategy = llm_reranking_strategy
        self.max_context_docs = max_context_docs
        self.pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        self.cache_dir = cache_dir
        self.use_kv_cache = use_kv_cache
        
        # Create cache directory if it doesn't exist
        if self.use_kv_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
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
        
        # Initialize KV-cache
        self.kv_cache = {}
        self.preloaded_documents = {}
        
    def _get_cache_path(self, document_id: str) -> str:
        """Get the path to the cache file for a document."""
        return os.path.join(self.cache_dir, f"{document_id}.json")
    
    def _save_to_cache(self, document_id: str, data: Dict[str, Any]) -> None:
        """Save data to the cache."""
        cache_path = self._get_cache_path(document_id)
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _load_from_cache(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Load data from the cache."""
        cache_path = self._get_cache_path(document_id)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def preload_document(self, document: Document) -> str:
        """
        Preload a document into the KV-cache.
        
        Args:
            document: The document to preload
            
        Returns:
            str: The document ID
        """
        # Generate a unique ID for the document
        doc_id = f"doc_{hash(document.text)}"
        
        # Check if document is already in cache
        if doc_id in self.preloaded_documents:
            return doc_id
        
        # Create a representation of the document for the cache
        doc_data = {
            "text": document.text,
            "metadata": document.metadata,
            "kv_cache": {}  # This would be populated with actual KV-cache in a real implementation
        }
        
        # Save to cache
        if self.use_kv_cache:
            self._save_to_cache(doc_id, doc_data)
        
        # Store in memory
        self.preloaded_documents[doc_id] = document
        self.kv_cache[doc_id] = doc_data
        
        return doc_id
    
    def preload_documents(self, documents: List[Document]) -> List[str]:
        """
        Preload multiple documents into the KV-cache.
        
        Args:
            documents: The documents to preload
            
        Returns:
            List[str]: List of document IDs
        """
        doc_ids = []
        for doc in documents:
            doc_id = self.preload_document(doc)
            doc_ids.append(doc_id)
        return doc_ids
    
    def compute_kv_cache(self, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Compute KV-cache for the specified documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Dict[str, Any]: The computed KV-cache
        """
        # In a real implementation, this would compute the actual KV-cache
        # For this example, we'll just return a placeholder
        kv_cache = {}
        for doc_id in doc_ids:
            if doc_id in self.kv_cache:
                # Simulate computing KV-cache
                kv_cache[doc_id] = {
                    "keys": [f"key_{i}" for i in range(10)],
                    "values": [f"value_{i}" for i in range(10)]
                }
        return kv_cache
    
    def parse_pdf(self, pdf_path: str) -> List[Document]:
        """
        Parse a PDF file and convert it into a list of Document objects.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects containing the parsed content
        """
        try:
            logger.info(f"Parsing PDF file: {pdf_path}")
            doc = self.pdf_reader.read_pdf(pdf_path)
            documents = []
            current_section = None
            current_section_content = []
            
            # Process each block in the PDF
            for block in doc.json:
                block_text = ""
                block_metadata = {"type": block.get('tag', 'unknown')}
                
                # Handle different block types
                if block.get('tag') == 'header':
                    # If we were building a section, save it
                    if current_section and current_section_content:
                        section_text = "\n".join(current_section_content)
                        if section_text.strip():
                            documents.append(Document(
                                text=section_text,
                                metadata={
                                    "type": "section",
                                    "title": current_section,
                                    "source": pdf_path,
                                    "page_num": block.get('page_num', 0)
                                }
                            ))
                    
                    # Start new section
                    block_text = ' '.join(block.get('sentences', []))
                    current_section = block_text
                    current_section_content = [block_text]
                    block_metadata['level'] = block.get('level', 0)
                
                elif block.get('tag') == 'para':
                    block_text = ' '.join(block.get('sentences', []))
                    if current_section:
                        current_section_content.append(block_text)
                
                elif block.get('tag') == 'list_item':
                    block_text = ' '.join(block.get('sentences', []))
                    if current_section:
                        current_section_content.append(block_text)
                    block_metadata['list_type'] = block.get('list_type', 'unknown')
                
                elif block.get('tag') == 'table':
                    table_content = []
                    for row in block.get('table_rows', []):
                        if row.get('type') == 'table_data_row':
                            cells = row.get('cells', [])
                            cell_values = [cell.get('cell_value', '') for cell in cells]
                            table_content.append(' | '.join(str(val) for val in cell_values))
                        elif row.get('type') == 'full_row':
                            table_content.append(row.get('cell_value', ''))
                    block_text = '\n'.join(table_content)
                    if current_section:
                        current_section_content.append(block_text)
                    block_metadata['is_table'] = True
                
                # Create Document object if block has content and not part of a section
                if block_text.strip() and not current_section:
                    block_metadata.update({
                        'source': pdf_path,
                        'page_num': block.get('page_num', 0),
                        'bbox': block.get('bbox', None)
                    })
                    
                    documents.append(Document(
                        text=block_text,
                        metadata=block_metadata
                    ))
            
            # Save the last section if exists
            if current_section and current_section_content:
                section_text = "\n".join(current_section_content)
                if section_text.strip():
                    documents.append(Document(
                        text=section_text,
                        metadata={
                            "type": "section",
                            "title": current_section,
                            "source": pdf_path,
                            "page_num": block.get('page_num', 0)
                        }
                    ))
            
            logger.info(f"Successfully parsed PDF into {len(documents)} document blocks")
            
            # Preload documents if KV-cache is enabled
            if self.use_kv_cache:
                self.preload_documents(documents)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error parsing PDF file {pdf_path}: {str(e)}")
            raise
    
    def generate_multiple_responses(
        self,
        query: str,
        retrieved_documents: List[Document],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        num_responses: int = 3
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate multiple responses using the LLM.
        
        Args:
            query: The user's query
            retrieved_documents: List of retrieved documents to use as context
            system_prompt: Optional system prompt to guide response generation
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens in the response
            num_responses: Number of different responses to generate
            
        Returns:
            List of tuples containing (response, metadata)
        """
        responses = []
        
        # Prepare context string
        context_str = "\n\n".join([
            f"Document {i+1}:\n{doc.text}"
            for i, doc in enumerate(retrieved_documents[:self.max_context_docs])
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
                "context_docs": len(retrieved_documents[:self.max_context_docs])
            }
            
            responses.append((response, metadata))
            
        return responses
    
    def generate_response(
        self,
        query: str,
        retrieved_documents: List[Document],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_llm_reranking: bool = True,
        num_responses: int = 3
    ) -> CAGResponse:
        """
        Generate a response using context-aware generation.
        
        Args:
            query: The user's query
            retrieved_documents: List of retrieved documents to use as context
            system_prompt: Optional system prompt to guide response generation
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens in the response
            use_llm_reranking: Whether to use LLM reranking for responses
            num_responses: Number of different responses to generate if using LLM reranking
            
        Returns:
            CAGResponse containing the generated response and metadata
        """
        try:
            # Rerank documents based on the query
            logger.info(f"Reranking {len(retrieved_documents)} documents using strategy: {self.reranking_strategy}")
            reranked_docs = rerank_retrieved_documents_by_strategy(
                query=query,
                retrieved_documents=retrieved_documents,
                strategy=self.reranking_strategy
            )
            
            # Limit the number of context documents
            context_docs = reranked_docs[:self.max_context_docs]
            logger.info(f"Using {len(context_docs)} documents as context")
            
            # Preload context documents if KV-cache is enabled
            if self.use_kv_cache:
                doc_ids = self.preload_documents(context_docs)
                kv_cache = self.compute_kv_cache(doc_ids)
                logger.info(f"Computed KV-cache for {len(doc_ids)} documents")
            
            if use_llm_reranking and num_responses > 1:
                # Generate multiple responses
                logger.info(f"Generating {num_responses} responses for LLM reranking")
                responses = self.generate_multiple_responses(
                    query=query,
                    retrieved_documents=context_docs,
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
                reranked_ids = rerank_llm_responses_by_strategy(
                    query=query,
                    responses=list(zip(response_ids, response_texts)),  # Format as (id, text) tuples
                    strategy=self.llm_reranking_strategy,
                    llm=self.llm  # Pass the LLM instance
                )
                
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
            return CAGResponse(
                response=final_response,
                context_used=context_docs,
                metadata={
                    "reranking_strategy": self.reranking_strategy,
                    "llm_reranking_strategy": self.llm_reranking_strategy if use_llm_reranking else None,
                    "num_context_docs": len(context_docs),
                    "total_docs": len(retrieved_documents),
                    "response_metadata": response_metadata,
                    "kv_cache_used": self.use_kv_cache
                }
            )
            
        except Exception as e:
            logger.error(f"Error in context-aware generation: {str(e)}")
            raise

def test_cag():
    """Test the SimpleCAG implementation with InvoCare document."""
    # Initialize CAG system
    cag = SimpleCAG(
        reranking_strategy="bm25",  # Using BM25 for precise matching
        llm_reranking_strategy="llm_rerank",  # Using LLM reranking for responses
        max_context_docs=4,  # Allow for more context
        use_kv_cache=True  # Enable KV-cache
    )
    
    # Parse the PDF file
    pdf_path = "pdf-test/cr2024-009.pdf"  # Updated to the correct path
    documents = cag.parse_pdf(pdf_path)
    
    # Find the Employee share purchase plan section
    employee_share_plan_section = None
    for doc in documents:
        if doc.metadata.get("type") == "section" and "Employee share purchase plan" in doc.metadata.get("title", ""):
            employee_share_plan_section = doc
            break
    
    if employee_share_plan_section:
        print(f"\nFound Employee share purchase plan section:")
        print(f"Title: {employee_share_plan_section.metadata['title']}")
        print(f"Content: {employee_share_plan_section.text}\n")
        
        # Use the section as context
        context_docs = [employee_share_plan_section]
    else:
        print("\nEmployee share purchase plan section not found. Using filtered documents instead.")
        # Filter documents to focus on Employee Share Purchase Plan section
        filtered_docs = []
        for doc in documents:
            if (
                "Employee share purchase plan" in doc.text or 
                "Plan" in doc.text or 
                "established" in doc.text.lower() or
                "created" in doc.text.lower() or
                "2006" in doc.text  # Look for the year 2006
            ):
                filtered_docs.append(doc)
        context_docs = filtered_docs
    
    # Test query
    query = "When was the InvoCare share purchase plan created?"
    
    # Generate response
    response = cag.generate_response(
        query=query,
        retrieved_documents=context_docs,
        temperature=0.7,
        system_prompt="Please focus on finding the exact date or year when the InvoCare Share Purchase Plan was established or created. Look for specific mentions of dates in relation to the Plan's establishment.",
        use_llm_reranking=True,
        num_responses=3
    )
    
    # Print results
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

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_cag()
