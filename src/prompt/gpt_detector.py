"""
GPT-based OOD detection using prompt engineering.
"""

import openai
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Optional
import json


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
        self.prompt_template = """
        You are an expert at detecting whether a question can be answered based on a given context.
        
        Context: {context}
        Question: {question}
        
        Please determine if the question can be answered using the provided context.
        
        Answer with only "YES" if the question can be answered from the context, or "NO" if it cannot.
        """
    
    def detect_ood(self, questions: List[str], contexts: List[str], 
                   batch_size: int = 10, delay: float = 1.0) -> List[float]:
        """
        Detect OOD questions using GPT.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            batch_size: Batch size for API calls
            delay: Delay between API calls (seconds)
            
        Returns:
            List of OOD scores (0 = ID, 1 = OOD)
        """
        scores = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc="GPT OOD Detection"):
            batch_questions = questions[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            batch_scores = []
            
            for question, context in zip(batch_questions, batch_contexts):
                try:
                    # Create prompt
                    prompt = self.prompt_template.format(
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
                    answer = response.choices[0].message.content.strip().upper()
                    
                    # Convert to score (NO = OOD = 1, YES = ID = 0)
                    score = 1.0 if answer == "NO" else 0.0
                    batch_scores.append(score)
                    
                    # Add delay to avoid rate limiting
                    time.sleep(delay)
                    
                except Exception as e:
                    print(f"Error processing question: {e}")
                    # Default to OOD if error occurs
                    batch_scores.append(1.0)
            
            scores.extend(batch_scores)
        
        return scores
    
    def save_results(self, scores: List[float], filepath: str):
        """
        Save detection results.
        
        Args:
            scores: OOD scores
            filepath: Path to save results
        """
        results = {
            'scores': scores,
            'model': self.model,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> List[float]:
        """
        Load detection results.
        
        Args:
            filepath: Path to results file
            
        Returns:
            List of OOD scores
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results['scores'] 