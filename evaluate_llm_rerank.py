import logging
import json
import time
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import jsonlines

from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from rerankers import (
    rerank_llm_responses_by_strategy,
    ReRankersInfo
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMRerankerEvaluator:
    def __init__(self, dataset_path: str = "retrievalqa.jsonl", num_samples: int = 50):
        """Initialize the LLM reranker evaluator.
        
        Args:
            dataset_path: Path to the retrievalqa.jsonl dataset
            num_samples: Number of samples to evaluate (use -1 for all samples)
        """
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        
        # Get all available strategies for LLM response reranking
        self.llm_reranking_strategies = ReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES.value
        logger.info(f"Available LLM response reranking strategies: {self.llm_reranking_strategies}")
        
        # Initialize T5 model for generation
        logger.info("Initializing T5 model")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        
        # Initialize OpenAI LLM for reranking
        try:
            self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
            logger.info("OpenAI LLM initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI LLM: {e}. Some reranking strategies may not work.")
            self.llm = None
        
        # Initialize metrics for each LLM response reranking strategy
        self.llm_reranking_metrics = {strategy: self._initialize_llm_reranking_metrics() for strategy in self.llm_reranking_strategies}
        # Add a no_reranking strategy for comparison
        self.llm_reranking_metrics["no_reranking"] = self._initialize_llm_reranking_metrics()
        
        # Load dataset
        self.test_cases = self._load_dataset()
        logger.info(f"Loaded {len(self.test_cases)} test cases from {dataset_path}")
        
    def _load_dataset(self) -> List[Dict]:
        """Load test cases from the retrievalqa.jsonl dataset."""
        test_cases = []
        try:
            with jsonlines.open(self.dataset_path) as reader:
                for item in reader:
                    # Convert the item format to our test case format
                    test_case = {
                        "question": item["question"],
                        "ground_truth": item["ground_truth"],
                        "contexts": [ctx["text"] for ctx in item["context"] if "text" in ctx]
                    }
                    test_cases.append(test_case)
                    
                    # Break if we've reached the desired number of samples
                    if self.num_samples > 0 and len(test_cases) >= self.num_samples:
                        break
                        
            return test_cases
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
            
    def _initialize_llm_reranking_metrics(self):
        """Initialize metrics dictionary for LLM response reranking."""
        return {
            "exact_match": [],
            "f1": [],
            "reranking_time": [],   # Time to rerank responses
        }
        
    def evaluate_generation(self, generated_answer: str, ground_truth_answers: List[str]) -> Dict[str, float]:
        """Evaluate answer generation performance.
        
        Args:
            generated_answer: The model-generated answer
            ground_truth_answers: The list of possible ground truth answers
            
        Returns:
            Dictionary of generation metrics
        """
        # Normalize answers
        generated_norm = generated_answer.lower().strip()
        
        # Check for exact match with any ground truth answer
        exact_match = any(generated_norm == gt.lower().strip() for gt in ground_truth_answers)
        
        # F1 Score calculation
        def get_tokens(text):
            return set(text.lower().split())
            
        pred_tokens = get_tokens(generated_answer)
        
        # Calculate F1 against each ground truth and take the best score
        best_f1 = 0.0
        for gt in ground_truth_answers:
            truth_tokens = get_tokens(gt)
            
            if not pred_tokens or not truth_tokens:
                f1 = 0.0
            else:
                precision = len(pred_tokens & truth_tokens) / len(pred_tokens)
                recall = len(pred_tokens & truth_tokens) / len(truth_tokens)
                current_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                best_f1 = max(best_f1, current_f1)
            
        return {
            "exact_match": float(exact_match),
            "f1": best_f1
        }
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using T5.
        
        Args:
            query: The question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        input_text = f"question: {query} context: {context}"
        input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
        
        outputs = self.model.generate(
            input_ids,
            max_length=150,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def evaluate_llm_response_reranking(self) -> Dict[str, Dict[str, float]]:
        """Evaluate LLM response reranking strategies."""
        if not self.llm:
            logger.error("OpenAI LLM not initialized. Cannot evaluate LLM response reranking.")
            return {}
            
        if not self.test_cases:
            logger.error("No test cases loaded. Cannot evaluate.")
            return {}
        
        logger.info(f"Starting LLM response reranking evaluation on {len(self.test_cases)} test cases")
        
        # Add storage for responses and evaluations for better analysis
        all_responses = {}
        all_evaluations = {}
        
        for i, test_case in enumerate(tqdm(self.test_cases)):
            try:
                query = test_case["question"]
                ground_truth_answers = test_case["ground_truth"]
                contexts = test_case["contexts"]
                
                # Store the query
                all_responses[query] = {}
                all_evaluations[query] = {}
                
                # Convert contexts to Document objects
                documents = [
                    Document(text=ctx, metadata={"id": f"doc_{j}"})
                    for j, ctx in enumerate(contexts)
                ]
                
                # Generate multiple responses with different contexts
                responses = self._generate_multiple_responses(query, documents)
                
                # Save all generated responses
                all_responses[query]["generated_responses"] = responses
                
                # If we couldn't generate multiple responses, skip this test case
                if len(responses) < 2:
                    logger.warning(f"Skipping test case {i}: Could not generate multiple responses")
                    continue
                
                # First evaluate the responses without reranking
                response_dict = {f"response_{j}": resp for j, resp in enumerate(responses)}
                no_reranking_answer = responses[0]  # Just take the first response
                no_reranking_metrics = self.evaluate_generation(no_reranking_answer, ground_truth_answers)
                
                # Store the evaluation for no_reranking
                all_evaluations[query]["no_reranking"] = {
                    "answer": no_reranking_answer,
                    "metrics": no_reranking_metrics
                }
                
                # Add metrics for no_reranking
                for metric, value in no_reranking_metrics.items():
                    if metric in self.llm_reranking_metrics["no_reranking"]:
                        self.llm_reranking_metrics["no_reranking"][metric].append(value)
                
                # Evaluate each LLM reranking strategy
                for strategy in self.llm_reranking_strategies:
                    try:
                        # Format responses for reranking
                        resp_tuples = [(f"response_{j}", resp) for j, resp in enumerate(responses)]
                        
                        # Measure reranking time
                        start_time = time.time()
                        
                        # Rerank responses using current strategy
                        reranked_ids = rerank_llm_responses_by_strategy(
                            query=query,
                            responses=resp_tuples,
                            strategy=strategy,
                            llm=self.llm
                        )
                        
                        # Calculate reranking time
                        reranking_time = time.time() - start_time
                        
                        # Get the top-ranked response
                        if reranked_ids and reranked_ids[0] in response_dict:
                            best_response = response_dict[reranked_ids[0]]
                            best_response_id = reranked_ids[0]
                        else:
                            logger.warning(f"Reranking with {strategy} returned invalid ID. Using first response.")
                            best_response = responses[0]
                            best_response_id = "response_0"
                        
                        # Evaluate the reranked response
                        generation_metrics = self.evaluate_generation(best_response, ground_truth_answers)
                        
                        # Store the evaluation for this strategy
                        all_evaluations[query][strategy] = {
                            "answer": best_response,
                            "metrics": generation_metrics,
                            "reranking_time": reranking_time,
                            "reranked_order": reranked_ids,
                            "best_response_id": best_response_id
                        }
                        
                        # Update metrics for this strategy
                        for metric, value in generation_metrics.items():
                            if metric in self.llm_reranking_metrics[strategy]:
                                self.llm_reranking_metrics[strategy][metric].append(value)
                        
                        # Add reranking time
                        self.llm_reranking_metrics[strategy]["reranking_time"].append(reranking_time)
                        
                        # Print information about this example
                        logger.info(f"Test case {i+1}, Strategy: {strategy}")
                        logger.info(f"  Query: {query}")
                        logger.info(f"  Ground truth: {ground_truth_answers}")
                        logger.info(f"  Best response: {best_response[:100]}...")
                        logger.info(f"  Selected: {best_response_id} (Ranking: {reranked_ids})")
                        logger.info(f"  Exact Match: {generation_metrics.get('exact_match', 0.0)}")
                        logger.info(f"  F1: {generation_metrics.get('f1', 0.0):.4f}")
                        logger.info(f"  Reranking Time: {reranking_time:.4f} sec")
                        
                    except Exception as e:
                        logger.error(f"Error evaluating LLM reranking strategy {strategy}: {str(e)}")
                        continue
                    
            except Exception as e:
                logger.error(f"Error processing test case {i}: {str(e)}")
                continue
        
        # Save detailed evaluation results
        with open("llm_reranking_detailed_results.json", "w") as f:
            json.dump(all_evaluations, f, indent=2)
                
        # Calculate final average metrics for each strategy
        final_results = {}
        for strategy, metrics in self.llm_reranking_metrics.items():
            final_results[strategy] = {
                metric: np.mean(values) if values else 0.0
                for metric, values in metrics.items()
            }
            
        return final_results
        
    def _generate_multiple_responses(self, query: str, documents: List[Document], num_responses: int = 3) -> List[str]:
        """Generate multiple different responses for the same query using different contexts."""
        responses = []
        
        try:
            # Generate responses with different contexts
            if len(documents) >= num_responses:
                # If we have enough documents, use different ones for each response
                for i in range(num_responses):
                    # Use a single document as context
                    context = documents[i].text
                    response = self.generate_answer(query, context)
                    responses.append(response)
            else:
                # If we don't have enough documents, vary the context in other ways
                # 1. First response: Use all documents
                all_context = "\n".join([doc.text for doc in documents])
                responses.append(self.generate_answer(query, all_context))
                
                # 2. Second response: Use partial context
                if len(documents) > 0:
                    partial_context = documents[0].text
                    responses.append(self.generate_answer(query, partial_context))
                
                # 3. Third response: No context
                responses.append(self.generate_answer(query, ""))
                
            # If we still need more responses, generate different variants by changing the prompt
            while len(responses) < num_responses:
                # Add some variation to the query to get different responses
                variant_query = f"Given the question '{query}', provide a comprehensive answer:"
                context = "\n".join([doc.text for doc in documents[:1]])
                responses.append(self.generate_answer(variant_query, context))
                
            return responses
            
        except Exception as e:
            logger.error(f"Error generating multiple responses: {str(e)}")
            # Return at least one response
            try:
                context = "\n".join([doc.text for doc in documents[:1]]) if documents else ""
                return [self.generate_answer(query, context)]
            except:
                return ["Unable to generate a response for the given query."]

def main():
    # Initialize evaluator with 50 samples from the dataset
    evaluator = LLMRerankerEvaluator(num_samples=50)
    
    # Run evaluation for LLM response reranking strategies
    logger.info("\nEvaluating LLM response reranking strategies...")
    llm_reranking_results = evaluator.evaluate_llm_response_reranking()
    
    # Save LLM reranking results
    with open("llm_reranking_evaluation_results.json", "w") as f:
        json.dump(llm_reranking_results, f, indent=4)
    
    # Print final LLM reranking results
    logger.info("\nLLM Response Reranking Final Results:")
    for strategy, metrics in llm_reranking_results.items():
        logger.info(f"\nStrategy: {strategy}")
        logger.info(f"  Exact Match: {metrics.get('exact_match', 0.0):.4f}")
        logger.info(f"  F1: {metrics.get('f1', 0.0):.4f}")
        logger.info(f"  Reranking Time: {metrics.get('reranking_time', 0.0):.4f} sec")

if __name__ == "__main__":
    main() 