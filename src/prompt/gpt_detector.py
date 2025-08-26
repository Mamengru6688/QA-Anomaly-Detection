"""
GPT-based OOD detection using prompt engineering.
Includes zero-shot, few-shot, and chain-of-thought (CoT) methods.
"""

import openai
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import json
import re


class GPTDetector:
    """GPT-based OOD detection using prompt engineering."""
    
    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo'):
        """
        Initialize GPT detector.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
        # Zero-shot prompt template (simple version)
        self.zero_shot_template = """You are a classifier for answerability detection.
You will be given a context and a question.

Rules:
- If the question is irrelevant to the context (i.e., out-of-domain), output "0".
- If the question **can be answered clearly and uniquely** based **only on the context**, output "1".
- In all other cases — including uncertain, ambiguous, or partially answerable questions — output "0".
- You MUST choose one of the two values even if uncertain.
- Output ONLY one number: either "0" or "1". No explanation or other text.

Important: The only valid outputs are "0" or "1". Do not leave the answer blank.

Context: {context}
Question: {question}
Answer:"""
        
        # Few-shot prompt template with examples
        self.few_shot_template = """You are a classifier for answerability detection.
Given a context and a question, your task is to decide whether the question can be clearly and uniquely answered based only on the context. Follow the rules:

Rules:
- If the question is out-of-domain (i.e., irrelevant to the context), output "0".
- If the question is ambiguous, uncertain, or not clearly answered, output "0".
- If the question can be clearly and uniquely answered from the context, output "1".
- Always choose "0" or "1". No explanation.

Examples:
Context: The capital of France is Paris.
Question: What is the capital of France?
Answer: 1

Context: The capital of France is Paris.
Question: Who is the president of the USA?
Answer: 0

Context: The capital of France is Paris.
Question: Where did Napoleon die?
Answer: 0

Now, answer the following:

Context: {context}
Question: {question}
Answer:"""
        
        # Chain-of-Thought prompt template
        self.cot_template = """You are an expert assistant for answerability detection.
You will be given a context and a question. Your task is to determine whether the question can be **clearly and uniquely answered using only the given context**.

Please follow these steps:
1. First, check if the question is relevant to the context. If not, it's Out-of-Domain, output 0.
2. If the question can be fully, clearly, and uniquely answered based only on the context, output "1".
3. In all other cases — including partially answerable, ambiguous, or uncertain questions, output "0".
4. You may choose "0" if unsure.
5. Do NOT rely on external knowledge, only use the context.

Output format:
Think step by step.
Then output only one number: "0" or "1".

---

Example 1:
Context: The Eiffel Tower is located in Paris, France.
Question: Where is the Eiffel Tower located?
Answer: 
- The question is directly related to the context.
- The answer ("Paris, France") is clearly stated.
Final answer: 1

---

Example 2:
Context: The Great Wall was built to protect against invasions.
Question: Who was the emperor during the Qin dynasty?
Answer:
- The question is not answerable from the context. It requires outside knowledge.
Final answer: 0

---

Example 3:
Context: The city's temperature hit 105°F on August 1, 1999, the highest on record.
Question: What was the coldest day on record?
Answer:
- The question is relevant, but the context only gives the highest temperature, not the lowest.
Final answer: 0

---

Now it's your turn:

Context: {context}
Question: {question}
Answer:"""
    
    def detect_ood_zero_shot(self, questions: List[str], contexts: List[str], 
                            batch_size: int = 10, delay: float = 1.0) -> List[float]:
        """
        Detect OOD questions using zero-shot prompting.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            batch_size: Batch size for API calls
            delay: Delay between API calls (seconds)
            
        Returns:
            List of OOD scores (0 = ID, 1 = OOD)
        """
        return self._detect_with_prompt(questions, contexts, self.zero_shot_template, 
                                      batch_size, delay, "Zero-shot GPT Detection")
    
    def detect_ood_few_shot(self, questions: List[str], contexts: List[str], 
                           batch_size: int = 10, delay: float = 1.0) -> List[float]:
        """
        Detect OOD questions using few-shot prompting.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            batch_size: Batch size for API calls
            delay: Delay between API calls (seconds)
            
        Returns:
            List of OOD scores (0 = ID, 1 = OOD)
        """
        return self._detect_with_prompt(questions, contexts, self.few_shot_template, 
                                      batch_size, delay, "Few-shot GPT Detection")
    
    def detect_ood_cot(self, questions: List[str], contexts: List[str], 
                      batch_size: int = 10, delay: float = 1.0) -> List[Tuple[float, str]]:
        """
        Detect OOD questions using chain-of-thought prompting.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            batch_size: Batch size for API calls
            delay: Delay between API calls (seconds)
            
        Returns:
            List of tuples (OOD scores, reasoning)
        """
        results = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc="CoT GPT Detection"):
            batch_questions = questions[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            batch_results = []
            
            for question, context in zip(batch_questions, batch_contexts):
                try:
                    # Create CoT prompt
                    prompt = self.cot_template.format(
                        context=context if context else "No context provided",
                        question=question
                    )
                    
                    # Call GPT API with more tokens for reasoning
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that thinks step by step."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0
                    )
                    
                    # Parse response
                    full_response = response.choices[0].message.content.strip()
                    
                    # Extract reasoning and final answer
                    reasoning = full_response
                    final_answer = self._extract_final_answer(full_response)
                    
                    # Convert to score (0 = ID, 1 = OOD)
                    score = 1.0 if final_answer == "0" else 0.0
                    batch_results.append((score, reasoning))
                    
                    # Add delay to avoid rate limiting
                    time.sleep(delay)
                    
                except Exception as e:
                    print(f"Error processing question: {e}")
                    # Default to OOD if error occurs
                    batch_results.append((1.0, f"Error: {e}"))
            
            results.extend(batch_results)
        
        return results
    
    def _detect_with_prompt(self, questions: List[str], contexts: List[str], 
                           prompt_template: str, batch_size: int, delay: float, 
                           desc: str) -> List[float]:
        """
        Generic detection method using a specific prompt template.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            prompt_template: Prompt template to use
            batch_size: Batch size for API calls
            delay: Delay between API calls (seconds)
            desc: Description for progress bar
            
        Returns:
            List of OOD scores (0 = ID, 1 = OOD)
        """
        scores = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc=desc):
            batch_questions = questions[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            batch_scores = []
            
            for question, context in zip(batch_questions, batch_contexts):
                try:
                    # Create prompt
                    prompt = prompt_template.format(
                        context=context if context else "No context provided",
                        question=question
                    )
                    
                    # Call GPT API
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=10,
                        temperature=0
                    )
                    
                    # Parse response
                    answer = response.choices[0].message.content.strip()
                    
                    # Convert to score (0 = ID, 1 = OOD)
                    score = 1.0 if answer == "0" else 0.0
                    batch_scores.append(score)
                    
                    # Add delay to avoid rate limiting
                    time.sleep(delay)
                    
                except Exception as e:
                    print(f"Error processing question: {e}")
                    # Default to OOD if error occurs
                    batch_scores.append(1.0)
            
            scores.extend(batch_scores)
        
        return scores
    
    def _extract_final_answer(self, response: str) -> str:
        """
        Extract final 0/1 answer from CoT response.
        
        Args:
            response: Full response from GPT
            
        Returns:
            Final answer (0 or 1)
        """
        # Look for 0 or 1 in the response
        response_clean = response.strip()
        
        # Check for explicit 0/1 answers
        if "0" in response_clean and "1" not in response_clean:
            return "0"
        elif "1" in response_clean and "0" not in response_clean:
            return "1"
        elif "0" in response_clean and "1" in response_clean:
            # If both appear, check which comes last (more likely to be the final answer)
            zero_pos = response_clean.rfind("0")
            one_pos = response_clean.rfind("1")
            return "0" if zero_pos > one_pos else "1"
        else:
            # If no explicit 0/1, try to infer from reasoning
            negative_words = ["cannot", "not", "unable", "impossible", "lacks", "missing", "out-of-domain"]
            positive_words = ["can", "able", "possible", "available", "contains", "provides", "clearly"]
            
            response_lower = response.lower()
            negative_count = sum(1 for word in negative_words if word in response_lower)
            positive_count = sum(1 for word in positive_words if word in response_lower)
            
            return "0" if negative_count > positive_count else "1"
    
    def detect_ood(self, questions: List[str], contexts: List[str], 
                   method: str = 'zero_shot', batch_size: int = 10, delay: float = 1.0):
        """
        Detect OOD questions using specified method.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            method: Detection method ('zero_shot', 'few_shot', 'cot')
            batch_size: Batch size for API calls
            delay: Delay between API calls (seconds)
            
        Returns:
            Detection results (scores or tuples with reasoning)
        """
        if method == 'zero_shot':
            return self.detect_ood_zero_shot(questions, contexts, batch_size, delay)
        elif method == 'few_shot':
            return self.detect_ood_few_shot(questions, contexts, batch_size, delay)
        elif method == 'cot':
            return self.detect_ood_cot(questions, contexts, batch_size, delay)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'zero_shot', 'few_shot', or 'cot'")
    
    def save_results(self, results, filepath: str, method: str = 'zero_shot'):
        """
        Save detection results.
        
        Args:
            results: Detection results (scores or tuples with reasoning)
            filepath: Path to save results
            method: Detection method used
        """
        if method == 'cot':
            # For CoT, results are tuples (score, reasoning)
            scores = [r[0] for r in results]
            reasoning = [r[1] for r in results]
            save_data = {
                'scores': scores,
                'reasoning': reasoning,
                'method': method,
                'model': self.model,
                'timestamp': time.time()
            }
        else:
            # For zero-shot and few-shot, results are just scores
            save_data = {
                'scores': results,
                'method': method,
                'model': self.model,
                'timestamp': time.time()
            }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str, method: str = 'zero_shot'):
        """
        Load detection results.
        
        Args:
            filepath: Path to results file
            method: Detection method used
            
        Returns:
            Detection results (scores or tuples with reasoning)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if method == 'cot' and 'reasoning' in data:
            # Reconstruct tuples from scores and reasoning
            scores = data['scores']
            reasoning = data['reasoning']
            return list(zip(scores, reasoning))
        else:
            return data['scores'] 
