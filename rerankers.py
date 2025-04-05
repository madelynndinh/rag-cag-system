"""
Helps to Re-rank the retrieved documents, Re-Rank LLM Generated Responses.

"""
import asyncio
import heapq
from typing import List, Dict, Tuple, Union, Any
from enum import Enum

from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.postprocessor import SentenceTransformerRerank


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
    STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES = ["llm_reranker", "llm_rerank", "sentence_transformer"]
    DEFAULT_STRATEGY_FOR_RERANKING_LLM_RESPONSES = STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES[0]
    RESPONSE_RELEVANCE_THRESHOLD_FOR_LLM_GENERATED_RESPONSES = 0.5
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

async def _rate_response(
    llm: Any,
    query: str,
    response: str,
    document_id: str,
    prompt: PromptTemplate
) -> Tuple[str, Dict[str, float]]:
    try:
        formatted_prompt = prompt.format(query=query, response=response)
        logger.info(f"Rating {document_id}: {response[:50]}...")

        # Synchronous call for simplicity during debugging
        llm_response = llm.complete(formatted_prompt)
        result_text = llm_response.text
        logger.info(f"Raw LLM response for {document_id}: {result_text}")

        # Parse JSON
        import json
        import re
        json_match = re.search(r'\{.*"rating".*\}', result_text, re.DOTALL)
        if json_match:
            rating_dict = json.loads(json_match.group(0))
            rating = float(rating_dict.get("rating", 0.5))
        else:
            number_match = re.search(r'(\d+(\.\d+)?)', result_text)
            rating = float(number_match.group(1)) / 10 if number_match and float(number_match.group(1)) > 1 else 0.5
            rating_dict = {"rating": rating}
        
        rating_dict["rating"] = max(0, min(1, rating))
        logger.info(f"Parsed rating for {document_id}: {rating_dict['rating']}")
        return document_id, rating_dict
    except Exception as e:
        logger.error(f"Error rating {document_id}: {e}")
        return document_id, {"rating": 0.0}
    
# Re-Rank the LLM Generated Responses using LLM as a Criticiser
async def _rerank_llm_responses_by_using_llm(
    query: str, responses: List[Tuple[str, str]], **kwargs
) -> List[str]:
    llm = kwargs.get("llm")
    assert llm, "LLM required"

    prompt = PromptTemplate(
        template=f"You are a helpful assistant that rates responses based on their relevance, accuracy, completeness, and clarity.\n\n{LLM_RESPONSE_RATER_PROMPT}",
        prompt_type=PromptType.SIMPLE_INPUT
    )
    
    tasks = [_rate_response(llm, query, resp, doc_id, prompt) for doc_id, resp in responses]
    ranked_responses = await asyncio.gather(*tasks)
    
    # Sort by rating descending
    sorted_responses = sorted(ranked_responses, key=lambda x: x[1]["rating"], reverse=True)
    logger.info(f"Sorted ratings: {[(doc_id, rating['rating']) for doc_id, rating in sorted_responses]}")
    
    return [doc_id for doc_id, _ in sorted_responses]

# Re-Rank the LLM Generated Responses using LLMRerank
def _rerank_llm_responses_using_llm_rerank(
    query: str, responses: List[Tuple[str, str]], **kwargs
) -> List[str]:
    """
    Re-rank the LLM generated responses using LLMRerank.
    
    Args:
        query (str): User query.
        responses (List[Tuple[str, str]]): List of tuples containing (document_id, response).
        **kwargs: Additional arguments for LLMRerank.
        
    Returns:
        List[str]: List of document IDs in reranked order.
    """
    try:
        logger.info(f"Reranking {len(responses)} LLM responses using LLMRerank")
        
        # Extract LLM from kwargs
        llm = kwargs.get("llm")
        assert llm, "LLM required for LLMRerank"
        
        # Convert responses to NodeWithScore format
        nodes = []
        for doc_id, response in responses:
            node = TextNode(
                text=response,
                metadata={"document_id": doc_id}
            )
            nodes.append(NodeWithScore(node=node, score=1.0))
        
        # Create LLMRerank postprocessor
        llm_rerank = LLMRerank(
            llm=llm,
            top_n=len(responses)  # Rerank all responses
        )
        
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)
        
        # Rerank nodes
        reranked_nodes = llm_rerank.postprocess_nodes(nodes, query_bundle)
        
        # Extract document IDs in order
        reranked_doc_ids = [node.node.metadata["document_id"] for node in reranked_nodes]
        
        logger.info(f"LLMRerank complete, returning {len(reranked_doc_ids)} responses")
        return reranked_doc_ids
    except Exception as e:
        logger.error(f"Error in re-ranking the LLM responses using LLMRerank: {str(e)}", exc_info=True)
        # Return original document IDs if reranking fails
        return [doc_id for doc_id, _ in responses[:ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value]]

# Re-Rank the LLM Generated Responses using SentenceTransformerRerank
def _rerank_llm_responses_using_sentence_transformer(
    query: str, responses: List[Tuple[str, str]], **kwargs
) -> List[str]:
    """
    Re-rank the LLM generated responses using SentenceTransformerRerank.
    
    Args:
        query (str): User query.
        responses (List[Tuple[str, str]]): List of tuples containing (document_id, response).
        **kwargs: Additional arguments for SentenceTransformerRerank.
        
    Returns:
        List[str]: List of document IDs in reranked order.
    """
    try:
        logger.info(f"Reranking {len(responses)} LLM responses using SentenceTransformerRerank")
        
        # Convert responses to NodeWithScore format
        nodes = []
        for doc_id, response in responses:
            node = TextNode(
                text=response,
                metadata={"document_id": doc_id}
            )
            nodes.append(NodeWithScore(node=node, score=1.0))
        
        # Create SentenceTransformerRerank postprocessor
        sentence_transformer = SentenceTransformerRerank(
            top_n=ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value,
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Updated to use correct model ID
        )
        
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)
        
        # Rerank nodes
        reranked_nodes = sentence_transformer.postprocess_nodes(nodes, query_bundle)
        
        # Extract document IDs in order
        reranked_doc_ids = [node.node.metadata["document_id"] for node in reranked_nodes]
        
        logger.info(f"SentenceTransformerRerank complete, returning {len(reranked_doc_ids)} responses")
        return reranked_doc_ids
    except Exception as e:
        logger.error(f"Error in re-ranking using SentenceTransformerRerank: {str(e)}", exc_info=True)
        # Return original document IDs if reranking fails
        return [doc_id for doc_id, _ in responses[:ReRankersInfo.TOP_K_DOCUMENTS_TO_RETURN.value]]

# Re-Rank the LLM Generated Responses by Strategy
def rerank_llm_responses_by_strategy(
    query: str, responses: List[Tuple[str, str]], strategy: str = None, **kwargs
) -> List[str]:
    try:
        logger.info("Starting reranking process")
        if not responses:
            logger.warning("No responses provided to re-rank.")
            return []
        elif len(responses) == 1:
            logger.info("Only one response provided. Returning as is.")
            return [responses[0][0]]
        
        if strategy is None:
            strategy = ReRankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_LLM_RESPONSES.value
            logger.info(f"Using default strategy: {strategy}")

        available_strategies = ReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES.value
        assert strategy in available_strategies, f"Invalid strategy. Available: {available_strategies}"
        logger.info(f"Strategy validated: {strategy}")

        reranked_responses = []
        if strategy == "llm_reranker" or strategy == "llm_rerank":  # Support both variants
            logger.info("Calling _rerank_llm_responses_using_llm_rerank")
            reranked_responses = _rerank_llm_responses_using_llm_rerank(
                query=query, responses=responses, **kwargs
            )
            logger.info(f"Reranked responses: {reranked_responses}")
        elif strategy == "sentence_transformer":
            logger.info("Calling _rerank_llm_responses_using_sentence_transformer")
            reranked_responses = _rerank_llm_responses_using_sentence_transformer(
                query=query, responses=responses, **kwargs
            )
            logger.info(f"Reranked responses: {reranked_responses}")
        return reranked_responses
    except Exception as e:
        logger.error(f"Error in reranking: {str(e)}", exc_info=True)
        return []
