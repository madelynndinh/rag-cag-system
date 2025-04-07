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
        RESPONSE_RELEVANCE_THRESHOLD_FOR_LLM_GENERATED_RESPONSES (float): Response relevance threshold for LLM generated responses (0-1).
    """
    STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES = ["llm_reranker", "llm_rerank", "sentence_transformer", "ensemble"]
    DEFAULT_STRATEGY_FOR_RERANKING_LLM_RESPONSES = STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES[0]
    SIMILARITY_THRESHOLD = 0.6

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


def _generate_multiple_responses(query: str, documents: List[Document], **kwargs) -> List[Tuple[str, str]]:
    """
    Generate multiple responses using different document combinations.
    """
    try:
        llm = kwargs.get("llm")
        assert llm, "LLM required for response generation"
        
        responses = []
        doc_combinations = [
            (documents[0:1], "single"),  # Best single document
            (documents[0:2], "pair"),    # Top 2 documents
            (documents, "all")           # All documents
        ]
        
        for docs, combo_type in doc_combinations:
            context = "\n".join([doc.text for doc in docs])
            prompt = f"""
            Based on the following context, answer the question accurately and concisely.
            Focus on geographic accuracy and proper administrative hierarchy.
            
            Context: {context}
            Question: {query}
            
            Answer:"""
            
            response = llm.complete(prompt).text
            doc_id = f"response_{combo_type}"
            responses.append((doc_id, response))
            
        return responses
    except Exception as e:
        logger.error(f"Error generating multiple responses: {str(e)}", exc_info=True)
        return []

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
        
        # Format responses for the prompt
        formatted_responses = []
        for i, (doc_id, response) in enumerate(responses, 1):
            formatted_responses.append(f"{i}. {response}")
        context_str = "\n".join(formatted_responses)
        
        
        
        # Convert responses to NodeWithScore format
        nodes = []
        for i, (doc_id, response) in enumerate(responses):
            node = TextNode(
                text=response,
                metadata={
                    "document_id": doc_id,
                    "choice_id": str(i + 1),
                    "index": i + 1
                }
            )
            nodes.append(NodeWithScore(node=node, score=1.0))
        
        # Debug log the nodes
        logger.debug(f"LLMRerank input nodes: {nodes}")# Create query bundle
        # Create LLMRerank with geographic-specific configuration
        llm_rerank = LLMRerank(
            llm=llm,
            top_n=len(responses),            
        )
        # Create query bundle
        query_bundle = QueryBundle(query_str=query)      
        # Rerank nodes
        reranked_nodes = llm_rerank.postprocess_nodes(nodes, query_bundle)
        
        # Debug log the reranked nodes
        logger.debug(f"LLMRerank output nodes: {reranked_nodes}")
        
        if not reranked_nodes:
            logger.warning("LLMRerank returned no nodes. Using original document IDs.")
            return [doc_id for doc_id, _ in responses]
        
        # Extract document IDs in order
        reranked_doc_ids = [
            node.node.metadata["document_id"] 
            for node in reranked_nodes 
            if hasattr(node, 'node') and hasattr(node.node, 'metadata')
        ]
        
        logger.info(f"Enhanced LLMRerank complete, returning {len(reranked_doc_ids)} responses")
        return reranked_doc_ids
    except Exception as e:
        logger.error(f"Error in enhanced LLM reranking: {str(e)}", exc_info=True)
        return [doc_id for doc_id, _ in responses]

# Re-Rank the LLM Generated Responses using SentenceTransformerRerank
def _rerank_llm_responses_using_sentence_transformer(
    query: str, responses: List[Tuple[str, str]], **kwargs
) -> List[str]:
    """
    Improved sentence transformer reranking with better model and thresholds.
    """
    try:
        logger.info(f"Reranking {len(responses)} LLM responses using improved SentenceTransformerRerank")
        
        original_doc_ids = [doc_id for doc_id, _ in responses]
        
        # Convert responses to NodeWithScore format
        nodes = []
        for doc_id, response in responses:
            node = TextNode(
                text=response,
                metadata={"document_id": doc_id}
            )
            nodes.append(NodeWithScore(node=node, score=1.0))
        
        # Create improved SentenceTransformerRerank
        sentence_transformer = SentenceTransformerRerank(
            top_n=len(responses),
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # Create query bundle with geographic context
        query_bundle = QueryBundle(
            query_str=f"{query}",
            custom_embedding_strs=[query]
        )
        
        # Rerank nodes
        reranked_nodes = sentence_transformer.postprocess_nodes(nodes, query_bundle)
        
        if not reranked_nodes:
            logger.warning("SentenceTransformerRerank returned no nodes. Using original document IDs.")
            return original_doc_ids
        
        # Extract document IDs with validation
        reranked_doc_ids = []
        for node in reranked_nodes:
            try:
                doc_id = node.node.metadata.get("document_id")
                if doc_id and doc_id in original_doc_ids:
                    reranked_doc_ids.append(doc_id)
            except Exception as e:
                logger.warning(f"Error extracting document ID: {e}")
                continue
        
        if not reranked_doc_ids:
            logger.warning("No valid document IDs extracted. Using original document IDs.")
            return original_doc_ids
        
        logger.info(f"Improved SentenceTransformerRerank complete, returning {len(reranked_doc_ids)} responses")
        return reranked_doc_ids
    except Exception as e:
        logger.error(f"Error in improved sentence transformer reranking: {str(e)}", exc_info=True)
        return original_doc_ids

# Re-Rank the LLM Generated Responses by Strategy
def rerank_llm_responses_by_strategy(
    query: str, responses: List[Tuple[str, str]], strategy: str = None, **kwargs
) -> List[str]:
    """
    Enhanced reranking with support for ensemble method.
    """
    try:
        logger.info("Starting reranking process")
        if not responses:
            logger.warning("No responses provided to re-rank.")
            return []
        elif len(responses) == 1:
            logger.info("Only one response provided. Returning as is.")
            return [responses[0][0]]
        
        original_doc_ids = [doc_id for doc_id, _ in responses]
        
        available_strategies = ReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES.value
        if strategy not in available_strategies:
            logger.warning(f"Invalid strategy. Using default. Available: {available_strategies}")
            strategy = ReRankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_LLM_RESPONSES.value
        
        logger.info(f"Strategy validated: {strategy}")
        
        if strategy == "ensemble":
            reranked_responses = _ensemble_rerank(query, responses, **kwargs)
        elif strategy in ["llm_reranker", "llm_rerank"]:
            reranked_responses = _rerank_llm_responses_using_llm_rerank(query, responses, **kwargs)
        elif strategy == "sentence_transformer":
            reranked_responses = _rerank_llm_responses_using_sentence_transformer(query, responses, **kwargs)
        else:
            logger.warning(f"Unknown strategy {strategy}. Using original order.")
            return original_doc_ids
            
        # Validate returned responses
        if not reranked_responses:
            logger.warning(f"Strategy {strategy} returned no responses. Using original document IDs.")
            return original_doc_ids
            
        # Ensure all returned IDs are valid
        valid_reranked_responses = [
            doc_id for doc_id in reranked_responses 
            if doc_id in set(original_doc_ids)
        ]
        
        if not valid_reranked_responses:
            logger.warning(f"No valid document IDs returned by {strategy}. Using original document IDs.")
            return original_doc_ids
            
        return valid_reranked_responses
    except Exception as e:
        logger.error(f"Error in reranking: {str(e)}", exc_info=True)
        return original_doc_ids

def _combine_rankings(rankings: List[List[str]]) -> List[str]:
    """
    Combine multiple rankings using Borda count method.
    """
    try:
        if not rankings:
            return []
            
        # Initialize score dictionary
        scores = {}
        n_rankings = len(rankings)
        n_items = len(rankings[0])
        
        # Calculate Borda scores
        for ranking in rankings:
            for position, doc_id in enumerate(ranking):
                if doc_id not in scores:
                    scores[doc_id] = 0
                # Higher positions get higher scores
                scores[doc_id] += (n_items - position)
        
        # Sort by scores
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items]
    except Exception as e:
        logger.error(f"Error combining rankings: {str(e)}", exc_info=True)
        return rankings[0] if rankings else []

def _ensemble_rerank(
    query: str, responses: List[Tuple[str, str]], **kwargs
) -> List[str]:
    """
    Ensemble reranking combining multiple strategies.
    """
    try:
        logger.info("Starting ensemble reranking")
        
        # Get rankings from different strategies
        llm_ranks = _rerank_llm_responses_using_llm_rerank(query, responses, **kwargs)
        st_ranks = _rerank_llm_responses_using_sentence_transformer(query, responses, **kwargs)
        
        # Combine rankings
        final_ranks = _combine_rankings([llm_ranks, st_ranks])
        
        logger.info(f"Ensemble reranking complete, returning {len(final_ranks)} responses")
        return final_ranks
    except Exception as e:
        logger.error(f"Error in ensemble reranking: {str(e)}", exc_info=True)
        return [doc_id for doc_id, _ in responses]