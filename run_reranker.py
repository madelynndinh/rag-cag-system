"""
Test script for running the reranker functionality.
"""
import json
import logging
import os
import time
from typing import Dict, List, Tuple, Set

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from rerankers import (
    _rerank_llm_responses_using_llm_rerank,
    _rerank_llm_responses_using_sentence_transformer,
    _combine_rankings,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data() -> List[Dict]:
    """Load test queries and their expected answers."""
    test_cases = [
        {
            "query": "In what country is Toronto Northwest?",
            "ground_truth": "Canada",
            "responses": [
                ("doc_1", "Toronto Northwest is located in Canada, specifically in the province of Ontario."),
                ("doc_2", "The area known as Toronto Northwest is situated in Canada."),
                ("doc_3", "Toronto Northwest is a region in Canada's largest city.")
            ]
        },
        {
            "query": "In what country is Lewałd Wielki?",
            "ground_truth": "Poland",
            "responses": [
                ("doc_1", "Lewałd Wielki is a village in Poland, located in the Warmian-Masurian Voivodeship."),
                ("doc_2", "The village of Lewałd Wielki is situated in northern Poland."),
                ("doc_3", "Lewałd Wielki belongs to the administrative district of Gmina Dąbrówno, within Poland.")
            ]
        }
    ]
    return test_cases

def evaluate_response(response: str, ground_truth: str) -> Tuple[float, float]:
    """
    Evaluate the response against ground truth using character-level n-grams.
    Returns (exact_match_score, f1_score)
    """
    # Extract the country name from the response
    response_lower = response.lower()
    ground_truth_lower = ground_truth.lower()
    
    # Check if the ground truth country is mentioned in the response
    exact_match = float(ground_truth_lower in response_lower)
    
    # Calculate F1 score based on character-level n-grams
    def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
        return set(text[i:i+n] for i in range(len(text)-n+1))
    
    response_ngrams = get_char_ngrams(response_lower)
    truth_ngrams = get_char_ngrams(ground_truth_lower)
    
    if not truth_ngrams:
        return exact_match, 0.0
    
    common_ngrams = response_ngrams.intersection(truth_ngrams)
    if not response_ngrams:
        return exact_match, 0.0
    
    precision = len(common_ngrams) / len(response_ngrams)
    recall = len(common_ngrams) / len(truth_ngrams)
    
    if precision + recall == 0:
        return exact_match, 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return exact_match, f1

def run_evaluation():
    """Run evaluation of different reranking strategies."""
    try:
        # Initialize LLM and settings
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
        Settings.llm = llm
        Settings.embed_model = OpenAIEmbedding()
        
        # Load test data
        test_cases = load_test_data()
        
        # Initialize results dictionary
        results = {
            "llm_reranker": {"exact_match": [], "f1": [], "time": []},
            "sentence_transformer": {"exact_match": [], "f1": [], "time": []},
            "ensemble": {"exact_match": [], "f1": [], "time": []},
            "no_reranking": {"exact_match": [], "f1": [], "time": []}
        }
        
        # Evaluate each test case
        for test_case in test_cases:
            query = test_case["query"]
            ground_truth = test_case["ground_truth"]
            responses = test_case["responses"]
            
            logger.info(f"\nEvaluating query: {query}")
            logger.info(f"Ground truth: {ground_truth}")
            
            # Test each reranking strategy
            for strategy in results.keys():
                start_time = time.time()
                
                try:
                    if strategy == "llm_reranker":
                        reranked_ids = _rerank_llm_responses_using_llm_rerank(
                            query, responses, llm=llm
                        )
                    elif strategy == "sentence_transformer":
                        reranked_ids = _rerank_llm_responses_using_sentence_transformer(
                            query, responses
                        )
                    elif strategy == "ensemble":
                        llm_ranks = _rerank_llm_responses_using_llm_rerank(
                            query, responses, llm=llm
                        )
                        st_ranks = _rerank_llm_responses_using_sentence_transformer(
                            query, responses
                        )
                        reranked_ids = _combine_rankings([llm_ranks, st_ranks])
                    else:  # no_reranking
                        reranked_ids = [doc_id for doc_id, _ in responses]
                    
                    # Get best response
                    best_response = next(
                        response for doc_id, response in responses 
                        if doc_id == reranked_ids[0]
                    )
                    
                    # Evaluate
                    exact_match, f1 = evaluate_response(best_response, ground_truth)
                    
                    # Record results
                    rerank_time = time.time() - start_time
                    results[strategy]["exact_match"].append(exact_match)
                    results[strategy]["f1"].append(f1)
                    results[strategy]["time"].append(rerank_time)
                    
                    logger.info(f"\n{strategy.upper()} Results:")
                    logger.info(f"Best response: {best_response}")
                    logger.info(f"Exact match: {exact_match}")
                    logger.info(f"F1 score: {f1:.4f}")
                    logger.info(f"Reranking time: {rerank_time:.4f}s")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {strategy}: {str(e)}", exc_info=True)
                    results[strategy]["exact_match"].append(0.0)
                    results[strategy]["f1"].append(0.0)
                    results[strategy]["time"].append(0.0)
        
        # Calculate averages
        final_results = {}
        for strategy in results:
            final_results[strategy] = {
                "avg_exact_match": sum(results[strategy]["exact_match"]) / len(test_cases),
                "avg_f1": sum(results[strategy]["f1"]) / len(test_cases),
                "avg_time": sum(results[strategy]["time"]) / len(test_cases)
            }
        
        # Save results
        with open("improved_reranking_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("\nFinal Results:")
        logger.info(json.dumps(final_results, indent=2))
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_evaluation()