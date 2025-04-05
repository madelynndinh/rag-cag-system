"""
Helps to Re-rank the retrieved documents, Re-Rank LLM Generated Responses.

"""
import asyncio
import heapq
from typing import List, Dict, Tuple, Union, Any
from enum import Enum

from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.retrievers import QueryFusionRetriever


from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
import logging



LOGGER_NAME = "rerankers"
logger = logging.getLogger(LOGGER_NAME)

class ReRankersInfo(Enum):
    """
    Hyperparameters required for reranking the documents or LLM Responses.

    Attributes:
        STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES (List[str]): Strategies available for reranking the LLM responses.
        DEFAULT_STRATEGY_FOR_RERANKING_LLM_RESPONSES (str): Selected strategy for reranking the LLM responses.
        RESPONSE_RELEVANCE_THRESHOLD_FOR_LLM_GENERATED_RESPONSES (float): Response relevance threshold for LLM generated responses (0-1).
        STRATEGIES_AVAILABLE_FOR_RERANKING_RETRIEVED_DOCUMENTS (List[str]): Strategies available for reranking the retrieved documents.
        DEFAULT_STRATEGY_FOR_RERANKING_RETRIEVED_DOCUMENTS (str): Selected strategy for reranking the retrieved documents.
        NO_OF_DOCUMENTS_TO_RERANK (int): Number of documents to rerank.
        TOP_K_DOCUMENTS_TO_RETURN (int): Number of documents to return after reranking.
    """
    STRATEGIES_AVAILABLE_FOR_RERANKING_RETRIEVED_DOCUMENTS = ["bm25", "queryfusion"]
    DEFAULT_STRATEGY_FOR_RERANKING_RETRIEVED_DOCUMENTS = STRATEGIES_AVAILABLE_FOR_RERANKING_RETRIEVED_DOCUMENTS[0]
    NO_OF_DOCUMENTS_TO_RERANK = 20  # Default number of documents to rerank
    TOP_K_DOCUMENTS_TO_RETURN = 4  # Only will return top-k documents after reranking


# ----------------------------- RE-RANKERS FOR LLM GENERATED RESPONSES -----------------------------
# Prompt to rate the responses, based on relevance and completeness and diversity of information.
LLM_RESPONSE_RATER_PROMPT = """
Given a user query and a response, rate the response on a scale of 0 to 1 based on:
1. Relevance to the query
2. Accuracy of information
3. Completeness of the answer
4. Clarity and coherence

Query: {query}
Response: {response}

Please provide your rating as a JSON object with a single field "rating" containing a float value between 0 and 1.
Example: {"rating": 0.85}

Rating:"""


# ----------------------------- RE-RANKERS FOR RETRIEVED DOCUMENTS BEFORE PASSING TO LLM -----------------------------

# Re-Rank the Retrieved Documents using BM25
def _rerank_retrieved_documents_using_bm25(
    query: str, retrieved_documents: List[Document], **kwargs
) -> List[Document]:
    """
    Re-rank the retrieved documents using BM25 algorithm.

    Args:
        query (str): User query.
        retrieved_documents (List[Document]): Retrieved documents.
        **kwargs: Additional arguments for BM25.

    Returns:
        List[Document]: Re-ranked retrieved documents based on BM25.
    """
    try:
        logger.info(f"Reranking {len(retrieved_documents)} documents using BM25")
        
        # Limit the number of documents to rerank
        docs_to_rerank = retrieved_documents[:ReRankersInfo.NO_OF_DOCUMENTS_TO_RERANK.value]
        logger.info(f"Reranking top {len(docs_to_rerank)} documents")
        
        # Convert documents to TextNode format with proper metadata
        nodes = []
        for i, doc in enumerate(docs_to_rerank):
            node = TextNode(
                text=doc.text,
                metadata={
                    "document_id": str(i),  # Use index as document ID
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "page_label": doc.metadata.get("page_label", "unknown"),
                }
            )
            nodes.append(node)
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value,
            # Use stemmer for better matching
            stemmer=Stemmer.Stemmer("english"),
            # Set language for stopwords
            language="english",
        )
        
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)
        
        # Retrieve and rerank nodes
        reranked_nodes = bm25_retriever.retrieve(query_bundle)
        
        # Extract documents in order, preserving original document metadata
        reranked_docs = []
        for node in reranked_nodes:
            # Find the original document that matches this node's text
            for doc in docs_to_rerank:
                if doc.text == node.text:
                    reranked_docs.append(doc)
                    break
        
        logger.info(f"BM25 reranking complete, returning {len(reranked_docs)} documents")
        return reranked_docs
    except Exception as e:
        logger.error(f"Error in re-ranking the retrieved documents using BM25: {str(e)}", exc_info=True)
        return retrieved_documents[:ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value]


# Re-Rank the Retrieved Documents using QueryFusionRetriever
def _rerank_retrieved_documents_using_queryfusion(
    query: str, retrieved_documents: List[Document], **kwargs
) -> List[Document]:
    """
    Re-rank the retrieved documents using QueryFusionRetriever.

    Args:
        query (str): User query.
        retrieved_documents (List[Document]): Retrieved documents.
        **kwargs: Additional arguments for QueryFusionRetriever.

    Returns:
        List[Document]: Re-ranked retrieved documents based on QueryFusionRetriever.
    """
    try:
        logger.info(f"Reranking {len(retrieved_documents)} documents using QueryFusionRetriever")
        
        # Limit the number of documents to rerank
        docs_to_rerank = retrieved_documents[:ReRankersInfo.NO_OF_DOCUMENTS_TO_RERANK.value]
        logger.info(f"Reranking top {len(docs_to_rerank)} documents")
        
        # Convert documents to TextNode format with proper metadata
        nodes = []
        for i, doc in enumerate(docs_to_rerank):
            node = TextNode(
                text=doc.text,
                metadata={
                    "document_id": str(i),  # Use index as document ID
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "page_label": doc.metadata.get("page_label", "unknown"),
                }
            )
            nodes.append(node)
        
        # Create vector retriever
        vector_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value,
            # Use stemmer for better matching
            stemmer=Stemmer.Stemmer("english"),
            # Set language for stopwords
            language="english",
        )
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value,
            # Use stemmer for better matching
            stemmer=Stemmer.Stemmer("english"),
            # Set language for stopwords
            language="english",
        )
        
        # Create QueryFusionRetriever
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value,
            num_queries=4,  # set this to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )
        
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)
        
        # Retrieve and rerank nodes
        reranked_nodes = fusion_retriever.retrieve(query_bundle)
        
        # Extract documents in order, preserving original document metadata
        reranked_docs = []
        for node in reranked_nodes:
            # Find the original document that matches this node's text
            for doc in docs_to_rerank:
                if doc.text == node.text:
                    reranked_docs.append(doc)
                    break
        
        logger.info(f"QueryFusionRetriever reranking complete, returning {len(reranked_docs)} documents")
        return reranked_docs
    except Exception as e:
        logger.error(f"Error in re-ranking the retrieved documents using QueryFusionRetriever: {str(e)}", exc_info=True)
        # Return original document IDs if reranking fails
        return retrieved_documents[:ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value]

# Re-Rank the Retrieved Documents by Strategy
def rerank_retrieved_documents_by_strategy(
    query: str, retrieved_documents: List[Document], strategy: str = None, **kwargs
) -> List[Document]:
    """
    Re-rank the retrieved documents based on the strategy provided.

    Args:
        query (str): User query.
        retrieved_documents (List[Document]): Retrieved documents.
        strategy (str): Strategy to re-rank the retrieved documents.
        kwargs: Additional arguments based on the strategy.

    Returns:
        reranked_documents (List[Document]): Re-ranked retrieved documents based on the strategy.
    """
    try:
        logger.info(f"Starting document reranking with {len(retrieved_documents)} documents")
        
        # Default strategy
        if strategy is None:
            strategy = ReRankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_RETRIEVED_DOCUMENTS.value
            logger.info(f"Using default strategy: {strategy}")

        # Validate the strategy
        available_strategies = ReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_RETRIEVED_DOCUMENTS.value
        assert strategy in available_strategies, f"Invalid strategy provided. Available strategies: {available_strategies}"
        logger.info(f"Strategy validated: {strategy}")

        # Re-rank the retrieved documents based on the strategy
        reranked_documents = []
        if strategy == "bm25":
            logger.info("Using BM25 reranking strategy")
            reranked_documents = _rerank_retrieved_documents_using_bm25(
                query=query, retrieved_documents=retrieved_documents, **kwargs
            )
        elif strategy == "queryfusion":
            logger.info("Using QueryFusionRetriever reranking strategy")
            reranked_documents = _rerank_retrieved_documents_using_queryfusion(
                query=query, retrieved_documents=retrieved_documents, **kwargs
            )
        elif strategy == "sentence_transformer":
            logger.info("Using SentenceTransformer reranking strategy")
            # This would be implemented in a similar way to the LLM response reranking
            # For now, we'll just return the original documents
            reranked_documents = retrieved_documents[:ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value]
        
        logger.info(f"Reranking complete, returning {len(reranked_documents)} documents")
        return reranked_documents
    except Exception as e:
        logger.error(f"Error in re-ranking the retrieved documents: {str(e)}", exc_info=True)
        # Return original documents if reranking fails
        return retrieved_documents[:ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value]
