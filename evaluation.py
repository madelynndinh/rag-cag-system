import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import json
import time
import os
from retrieval import (
    rerank_retrieved_documents_by_strategy,
    ReRankersInfo
)
from rerankers import (
    rerank_llm_responses_by_strategy,
    ReRankersInfo as LLMRerankersInfo
)
from parsing import parse_pdf
from llama_index.core import Document
from llama_index.llms.openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(
        self,
        dataset_path: str = "retrievalqa.jsonl",
        k_values: List[int] = [1, 3, 5, 10],
        rerankers_info: ReRankersInfo = None,
        llm_rerankers_info: LLMRerankersInfo = None,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """Initialize the RAG evaluator.
        
        Args:
            dataset_path: Path to the local dataset file
            k_values: List of k values for Recall@k and Precision@k
            rerankers_info: Configuration for reranking strategies
            llm_rerankers_info: Configuration for LLM response reranking strategies
            timeout: Timeout in seconds for dataset loading
            max_retries: Maximum number of retries for dataset loading
        """
        self.dataset_path = dataset_path
        self.k_values = k_values
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Get all available strategies for document retrieval
        self.reranking_strategies = ReRankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_RETRIEVED_DOCUMENTS.value
        logger.info(f"Available document reranking strategies: {self.reranking_strategies}")
        
        # Get all available strategies for LLM response reranking
        self.llm_reranking_strategies = LLMRerankersInfo.STRATEGIES_AVAILABLE_FOR_RERANKING_LLM_RESPONSES.value
        logger.info(f"Available LLM response reranking strategies: {self.llm_reranking_strategies}")
        
        # Use the default strategy from ReRankersInfo if none provided
        self.default_strategy = rerankers_info or ReRankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_RETRIEVED_DOCUMENTS.value
        
        # Use the default LLM reranking strategy if none provided
        self.default_llm_strategy = llm_rerankers_info or LLMRerankersInfo.DEFAULT_STRATEGY_FOR_RERANKING_LLM_RESPONSES.value
        
        # Initialize searcher for retrieving documents
        self.searcher = self._initialize_searcher()
        
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
        
        # Initialize metrics for each document retrieval strategy
        self.metrics_by_strategy = {strategy: self._initialize_metrics() for strategy in self.reranking_strategies}
        # Add a no_retrieval strategy for comparison
        self.metrics_by_strategy["no_retrieval"] = self._initialize_metrics()
        
        # Initialize metrics for each LLM response reranking strategy
        self.llm_reranking_metrics = {strategy: self._initialize_llm_reranking_metrics() for strategy in self.llm_reranking_strategies}
        # Add a no_reranking strategy for comparison
        self.llm_reranking_metrics["no_reranking"] = self._initialize_llm_reranking_metrics()
        
    def _initialize_metrics(self):
        """Initialize metrics dictionary."""
        metrics = {
            f"recall@{k}": [] for k in self.k_values
        }
        metrics.update({
            f"precision@{k}": [] for k in self.k_values
        })
        metrics.update({
            "exact_match": [],
            "f1": [],
            "retrieval_accuracy": [],  # Whether retrieval was correctly decided
        })
        return metrics

    def _initialize_searcher(self):
        """Initialize a searcher for retrieving documents."""
        try:
            # Try to initialize Pyserini searcher
            logger.info("Initializing LuceneSearcher")
            return None  # We'll use the context from the dataset for now
        except Exception as e:
            logger.error(f"Error initializing searcher: {e}")
            return None

    def _get_dataset_examples(self, num_samples: int = 100):
        """Load examples directly from the jsonl file without using Dataset class."""
        try:
            logger.info(f"Loading examples from {self.dataset_path}")
            
            if not os.path.exists(self.dataset_path):
                logger.warning(f"File {self.dataset_path} not found, checking in ./data/ directory")
                data_path = os.path.join("data", self.dataset_path)
                if not os.path.exists(data_path):
                    logger.error(f"Dataset file not found at {self.dataset_path} or {data_path}")
                    return self._get_sample_examples()
                self.dataset_path = data_path
            
            # Read JSONL file line by line
            examples = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    try:
                        example = json.loads(line.strip())
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON on line {i+1}: {e}")
                        continue
            
            logger.info(f"Successfully loaded {len(examples)} examples from local file")
            return examples
            
        except Exception as e:
            logger.error(f"Error loading examples from local file: {str(e)}")
            return self._get_sample_examples()
    
    def _get_sample_examples(self):
        """Create sample examples for testing."""
        logger.info("Creating sample examples for testing")
        
        return [
            {
                "question": "What percentage of couples are 'sleep divorced', according to new research?",
                "ground_truth": ["15%"],
                "param_knowledge_answerable": 0,
                "context": [{"title": "Sleep Research", "text": "15% of respondents have started a sleep divorce."}],
                "data_source": "realtimeqa",
                "question_id": "sample_q1"
            },
            {
                "question": "Who is the CEO of OpenAI as of 2023?",
                "ground_truth": ["Sam Altman"],
                "param_knowledge_answerable": 0,
                "context": [{"title": "OpenAI Leadership", "text": "Sam Altman returned as CEO of OpenAI in late 2023."}],
                "data_source": "freshqa",
                "question_id": "sample_q2"
            },
            {
                "question": "What is the capital of France?",
                "ground_truth": ["Paris"],
                "param_knowledge_answerable": 1,
                "context": [{"title": "France", "text": "Paris is the capital city of France."}],
                "data_source": "general",
                "question_id": "sample_q3"
            }
        ]

    def _prepare_documents_from_context(self, context_list):
        """Convert context from RetrievalQA format to Document objects."""
        documents = []
        if not context_list:
            return documents
            
        for i, ctx in enumerate(context_list):
            # Skip if ctx is None or not a valid dictionary
            if not ctx or not isinstance(ctx, dict):
                continue
                
            # Get text from context item
            text = ctx.get("text", "")
            if not text:
                continue
                
            doc = Document(
                text=text,
                metadata={
                    "id": f"doc_{i}",
                    "title": ctx.get("title", ""),
                    "source": "context"
                }
            )
            documents.append(doc)
        return documents
    
    def retrieve_documents(self, query: str, context_documents=None):
        """Retrieve documents for a query.
        If context_documents is provided, use those instead of dummy documents.
        """
        if context_documents and len(context_documents) >= 1:
            # If we have context documents, duplicate them to meet minimum requirements
            # for BM25 retriever which needs at least 4 documents
            docs = list(context_documents)
            
            # If we have fewer than 4 documents, duplicate the existing ones
            while len(docs) < 4:
                for doc in context_documents:
                    if len(docs) >= 4:
                        break
                    # Create a copy with slightly modified text to avoid exact duplicates
                    dup_doc = Document(
                        text=f"{doc.text} [duplicate]",
                        metadata={
                            "id": f"{doc.metadata.get('id', '')}_dup_{len(docs)}",
                            "title": doc.metadata.get("title", ""),
                            "source": "context_duplicate"
                        }
                    )
                    docs.append(dup_doc)
            return docs
        
   
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Document], reference_docs: List[Document]) -> Dict[str, float]:
        """Evaluate retrieval performance.
        
        Args:
            query: The search query
            retrieved_docs: List of retrieved documents
            reference_docs: List of reference documents from the dataset
            
        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {}
        retrieved_texts = [doc.text for doc in retrieved_docs]
        reference_texts = [doc.text for doc in reference_docs]
        
        # Calculate Recall@k
        for k in self.k_values:
            top_k_texts = retrieved_texts[:k] if k <= len(retrieved_texts) else retrieved_texts
            
            # Count how many reference documents were retrieved
            retrieved_count = sum(1 for ref_text in reference_texts if any(ref_text in ret_text for ret_text in top_k_texts))
            recall = retrieved_count / len(reference_texts) if reference_texts else 0
            metrics[f"recall@{k}"] = recall
            
            # Calculate Precision@k
            precision = retrieved_count / len(top_k_texts) if top_k_texts else 0
            metrics[f"precision@{k}"] = precision
            
        return metrics

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

    def evaluate(self, num_samples: int = 100, evaluate_retrieval_decision: bool = True) -> Dict[str, Dict[str, float]]:
        """Run the complete evaluation pipeline for all reranking strategies.
        
        Args:
            num_samples: Number of samples to evaluate
            evaluate_retrieval_decision: Whether to evaluate the accuracy of retrieval decisions
            
        Returns:
            Dictionary of strategies to average metrics
        """
        logger.info(f"Starting evaluation on up to {num_samples} samples for all reranking strategies")
        
        # Load examples directly from JSONL
        examples = self._get_dataset_examples(num_samples)
        actual_samples = len(examples)
        
        logger.info(f"Successfully loaded {actual_samples} examples for evaluation")
        
        for i, example in enumerate(tqdm(examples)):
            try:
                # Check if we have all required fields
                if "question" not in example:
                    logger.warning(f"Skipping example {i}: No question field")
                    continue
                
                query = example["question"]
                if not query:
                    logger.warning(f"Skipping example {i}: Empty question")
                    continue
                
                # Handle ground truth answers
                if "ground_truth" not in example or not example["ground_truth"]:
                    logger.warning(f"Skipping example {i}: No ground truth answers")
                    continue
                
                ground_truth_answers = example["ground_truth"]
                # If ground_truth is not a list or contains None values, fix it
                if not isinstance(ground_truth_answers, list):
                    ground_truth_answers = [ground_truth_answers]
                ground_truth_answers = [str(gt) for gt in ground_truth_answers if gt is not None]
                
                if not ground_truth_answers:
                    logger.warning(f"Skipping example {i}: No valid ground truth answers")
                    continue
                
                # Determine if retrieval is needed
                needs_retrieval = example.get("param_knowledge_answerable", 0) == 0
                
                # Get reference documents from the dataset
                context_data = example.get("context", [])
                # Make sure context is a list
                if not isinstance(context_data, list):
                    context_data = [context_data]
                # Filter out None values
                context_data = [ctx for ctx in context_data if ctx is not None]
                
                reference_docs = self._prepare_documents_from_context(context_data)
                
                if not reference_docs:
                    logger.warning(f"Example {i} has no valid reference documents. Creating dummy references.")
                    # Create at least one reference document based on the query
                    reference_docs = [
                        Document(
                            text=f"Reference information for: {query}",
                            metadata={"id": "dummy_ref", "source": "dummy_reference"}
                        )
                    ]
                
                # Use actual context documents from the dataset for retrieval
                initial_docs = self.retrieve_documents(query, reference_docs)
                
                # Generate answer without retrieval
                no_retrieval_answer = self.generate_answer(query, "")  # Empty context
                no_retrieval_metrics = self.evaluate_generation(no_retrieval_answer, ground_truth_answers)
                
                # Add to no_retrieval metrics
                for metric, value in no_retrieval_metrics.items():
                    self.metrics_by_strategy["no_retrieval"][metric].append(value)
                
                # Evaluate retrieval_accuracy only once (not dependent on reranking strategy)
                if evaluate_retrieval_decision:
                    # This would be the LLM's decision on whether to retrieve
                    retrieval_needed_prediction = self._should_retrieve(query)
                    retrieval_accuracy = float(retrieval_needed_prediction == needs_retrieval)
                    # Add retrieval accuracy to all strategies
                    for strategy in self.metrics_by_strategy:
                        self.metrics_by_strategy[strategy]["retrieval_accuracy"].append(retrieval_accuracy)
                
                # Evaluate each reranking strategy
                for strategy in self.reranking_strategies:
                    try:
                        # Rerank documents using the current strategy
                        retrieved_docs = rerank_retrieved_documents_by_strategy(
                            query=query,
                            retrieved_documents=initial_docs,
                            strategy=strategy
                        )
                    except Exception as e:
                        logger.warning(f"Reranking with {strategy} failed: {str(e)}. Using initial documents.")
                        # If reranking fails, just use the initial documents
                        retrieved_docs = initial_docs[:3]  # Take top 3
                    
                    # Evaluate retrieval
                    retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, reference_docs)
                    
                    # Generate answer with retrieval
                    context = "\n".join([doc.text for doc in retrieved_docs[:3]])  # Use top 3 docs
                    generated_answer = self.generate_answer(query, context)
                    
                    # Evaluate generation
                    generation_metrics = self.evaluate_generation(generated_answer, ground_truth_answers)
                    
                    # Update metrics for this strategy
                    for metric, value in {**retrieval_metrics, **generation_metrics}.items():
                        self.metrics_by_strategy[strategy][metric].append(value)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1} samples")
                    self._log_current_metrics()
            
            except Exception as e:
                logger.error(f"Error processing example {i}: {str(e)}")
                continue
        
        # Calculate final average metrics for each strategy
        final_results = {}
        for strategy, metrics in self.metrics_by_strategy.items():
            final_results[strategy] = {
                metric: np.mean(values) if values else 0.0
                for metric, values in metrics.items()
            }
        
        return final_results

    def _should_retrieve(self, query: str) -> bool:
        """Decide whether to retrieve for a query based on the model's prediction.
        This is a simplified version - in practice, you might use a more sophisticated approach.
        
        Args:
            query: The query to evaluate
            
        Returns:
            Boolean indicating whether retrieval is needed
        """
        # Add a simple prompt asking if retrieval is needed
        retrieval_prompt = f"Does answering the following question require looking up external information? Answer YES or NO.\nQuestion: {query}"
        input_ids = self.tokenizer(retrieval_prompt, return_tensors="pt", max_length=512).input_ids
        
        outputs = self.model.generate(
            input_ids,
            max_length=10,
            num_beams=2,
            early_stopping=True
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        return "yes" in answer

    def _log_current_metrics(self):
        """Log current average metrics for each strategy."""
        logger.info("Current metrics:")
        
        for strategy, metrics in self.metrics_by_strategy.items():
            current_metrics = {
                metric: np.mean(values) if values else 0.0
                for metric, values in metrics.items()
            }
            
            logger.info(f"Strategy: {strategy}")
            logger.info(f"  Exact Match: {current_metrics.get('exact_match', 0.0):.4f}")
            logger.info(f"  F1: {current_metrics.get('f1', 0.0):.4f}")
            logger.info(f"  Recall@3: {current_metrics.get('recall@3', 0.0):.4f}")

    def _initialize_llm_reranking_metrics(self):
        """Initialize metrics dictionary for LLM response reranking."""
        return {
            "exact_match": [],
            "f1": [],
            "relevance_score": [],  # Human-judged or proxy score for relevance
            "generation_time": [],  # Time to generate responses
            "reranking_time": [],   # Time to rerank responses
        }

    def evaluate_llm_response_reranking(self, num_samples: int = 20) -> Dict[str, Dict[str, float]]:
        """Evaluate LLM response reranking strategies.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary of strategies to average metrics
        """
        if not self.llm:
            logger.error("OpenAI LLM not initialized. Cannot evaluate LLM response reranking.")
            return {}
            
        logger.info(f"Starting LLM response reranking evaluation on {num_samples} samples")
        
        # Load examples
        examples = self._get_dataset_examples(num_samples)
        actual_samples = len(examples)
        
        logger.info(f"Successfully loaded {actual_samples} examples for evaluation")
        
        for i, example in enumerate(tqdm(examples[:num_samples])):
            try:
                # Check if we have all required fields
                if "question" not in example:
                    logger.warning(f"Skipping example {i}: No question field")
                    continue
                
                query = example["question"]
                if not query:
                    logger.warning(f"Skipping example {i}: Empty question")
                    continue
                
                # Handle ground truth answers
                if "ground_truth" not in example or not example["ground_truth"]:
                    logger.warning(f"Skipping example {i}: No ground truth answers")
                    continue
                
                ground_truth_answers = example["ground_truth"]
                # Fix ground truth if needed
                if not isinstance(ground_truth_answers, list):
                    ground_truth_answers = [ground_truth_answers]
                ground_truth_answers = [str(gt) for gt in ground_truth_answers if gt is not None]
                
                # Get reference documents
                context_data = example.get("context", [])
                # Make sure context is a list
                if not isinstance(context_data, list):
                    context_data = [context_data]
                # Filter out None values
                context_data = [ctx for ctx in context_data if ctx is not None]
                
                reference_docs = self._prepare_documents_from_context(context_data)
                
                if not reference_docs:
                    logger.warning(f"Example {i} has no valid reference documents. Creating dummy references.")
                    reference_docs = [
                        Document(
                            text=f"Reference information for: {query}",
                            metadata={"id": "dummy_ref", "source": "dummy_reference"}
                        )
                    ]
                
                # Generate multiple LLM responses with different contexts/prompts
                responses = self._generate_multiple_responses(query, reference_docs)
                
                # If we couldn't generate multiple responses, skip this example
                if len(responses) < 2:
                    logger.warning(f"Skipping example {i}: Could not generate multiple responses")
                    continue
                
                # First evaluate the responses without reranking
                response_dict = {f"response_{j}": resp for j, resp in enumerate(responses)}
                no_reranking_answer = responses[0]  # Just take the first response
                no_reranking_metrics = self.evaluate_generation(no_reranking_answer, ground_truth_answers)
                
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
                        else:
                            logger.warning(f"Reranking with {strategy} returned invalid ID. Using first response.")
                            best_response = responses[0]
                        
                        # Evaluate the reranked response
                        generation_metrics = self.evaluate_generation(best_response, ground_truth_answers)
                        
                        # Update metrics for this strategy
                        for metric, value in generation_metrics.items():
                            if metric in self.llm_reranking_metrics[strategy]:
                                self.llm_reranking_metrics[strategy][metric].append(value)
                        
                        # Add reranking time
                        self.llm_reranking_metrics[strategy]["reranking_time"].append(reranking_time)
                        
                    except Exception as e:
                        logger.error(f"Error evaluating LLM reranking strategy {strategy}: {str(e)}")
                        continue
                
                # Log progress
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i + 1} samples for LLM response reranking")
                    self._log_current_llm_reranking_metrics()
            
            except Exception as e:
                logger.error(f"Error processing example {i} for LLM response reranking: {str(e)}")
                continue
        
        # Calculate final average metrics for each strategy
        final_results = {}
        for strategy, metrics in self.llm_reranking_metrics.items():
            final_results[strategy] = {
                metric: np.mean(values) if values else 0.0
                for metric, values in metrics.items()
            }
        
        return final_results
        
    def _generate_multiple_responses(self, query: str, documents: List[Document], num_responses: int = 3) -> List[str]:
        """Generate multiple different responses for the same query using different contexts.
        
        Args:
            query: The question
            documents: List of available documents
            num_responses: Number of different responses to generate
            
        Returns:
            List of generated responses
        """
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
    
    def _log_current_llm_reranking_metrics(self):
        """Log current average metrics for each LLM reranking strategy."""
        logger.info("Current LLM reranking metrics:")
        
        for strategy, metrics in self.llm_reranking_metrics.items():
            current_metrics = {
                metric: np.mean(values) if values else 0.0
                for metric, values in metrics.items()
            }
            
            logger.info(f"Strategy: {strategy}")
            logger.info(f"  Exact Match: {current_metrics.get('exact_match', 0.0):.4f}")
            logger.info(f"  F1: {current_metrics.get('f1', 0.0):.4f}")
            logger.info(f"  Reranking Time: {current_metrics.get('reranking_time', 0.0):.4f} sec")

def main():
    # Initialize evaluator
    evaluator = RAGEvaluator(
        dataset_path="retrievalqa.jsonl",
        k_values=[1, 3, 5, 10],
        timeout=120,
        max_retries=3
    )
    
    # Run evaluation for document retrieval strategies
    logger.info("Evaluating document retrieval strategies...")
    retrieval_results = evaluator.evaluate(num_samples=20)  # Using fewer samples for initial testing
    
    # Save document retrieval results
    with open("evaluation_results_by_strategy.json", "w") as f:
        json.dump(retrieval_results, f, indent=4)
    
    # Print final document retrieval results
    logger.info("\nDocument Retrieval Final Results:")
    for strategy, metrics in retrieval_results.items():
        logger.info(f"\nStrategy: {strategy}")
        logger.info(f"  Exact Match: {metrics.get('exact_match', 0.0):.4f}")
        logger.info(f"  F1: {metrics.get('f1', 0.0):.4f}")
        
        # Print recall and precision metrics
        for k in [1, 3, 5, 10]:
            if f"recall@{k}" in metrics:
                logger.info(f"  Recall@{k}: {metrics.get(f'recall@{k}', 0.0):.4f}")
                logger.info(f"  Precision@{k}: {metrics.get(f'precision@{k}', 0.0):.4f}")
                
        if "retrieval_accuracy" in metrics:
            logger.info(f"  Retrieval Decision Accuracy: {metrics.get('retrieval_accuracy', 0.0):.4f}")
    
    # Run evaluation for LLM response reranking strategies
    logger.info("\nEvaluating LLM response reranking strategies...")
    llm_reranking_results = evaluator.evaluate_llm_response_reranking(num_samples=10)  # Using even fewer samples
    
    # Save LLM reranking results
    with open("evaluation_results_llm_reranking.json", "w") as f:
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