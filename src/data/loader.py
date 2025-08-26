"""
Data loader module for anomaly detection in QA systems.
Handles loading and preprocessing of SQuAD v2 and TriviaQA datasets.
Based on original ad.py implementation.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import random
from typing import Tuple, Dict, List


class QADataLoader:
    """Data loader for question-answering datasets based on ad.py implementation."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data loader.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def load_squad_v2_data(self, split: str = 'train') -> pd.DataFrame:
        """
        Load SQuAD v2 dataset (matching ad.py implementation).
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            
        Returns:
            DataFrame with questions and contexts
        """
        dataset = load_dataset('squad_v2', split=split)
        
        # Extract questions and contexts
        data = []
        for item in dataset:
            data.append({
                'question': item['question'],
                'context': item['context'],
                'answer': item['answers']['text'][0] if item['answers']['text'] else '',
                'label_ID': 1,  # In-distribution
                'label_A': 1 if len(item['answers']['text']) > 0 else 0,  # Answerability label
                'label': 1 if len(item['answers']['text']) > 0 else 0  # Final label (ID-A = 1, ID-U = 0)
            })
        
        return pd.DataFrame(data)
    
    def load_trivia_data(self, split: str = 'train') -> pd.DataFrame:
        """
        Load TriviaQA dataset (matching ad.py implementation).
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            
        Returns:
            DataFrame with questions
        """
        dataset = load_dataset('trivia_qa', 'unfiltered', split=split)
        
        # Extract questions (matching ad.py logic)
        data = []
        for item in dataset:
            data.append({
                'question': item['question'],
                'context': '',  # TriviaQA doesn't have contexts
                'answer': item['answer']['value'] if 'answer' in item else '',
                'label_ID': 0,  # Out-of-distribution
                'label_A': 0,   # Unanswerable (OOD)
                'label': 0      # Final label (OOD = 0)
            })
        
        return pd.DataFrame(data)
    
    def create_full_dataset(self, 
                           id_a_samples: int = 8000,
                           id_u_samples: int = 8000,
                           ood_u_samples: int = 5000) -> pd.DataFrame:
        """
        Create full dataset with ID-A, ID-U, and OOD-U samples (matching ad.py).
        
        Args:
            id_a_samples: Number of ID-A (answerable) samples
            id_u_samples: Number of ID-U (unanswerable) samples  
            ood_u_samples: Number of OOD-U (out-of-domain) samples
            
        Returns:
            Combined dataset with all sample types
        """
        # Load SQuAD v2 data
        squad = load_dataset("squad_v2", split="train")
        
        # Load TriviaQA data
        trivia = load_dataset("trivia_qa", "unfiltered", split="train")
        
        # Extract answerable samples (ID-A)
        squad_answerable = [ex for ex in squad if len(ex["answers"]["text"]) > 0]
        id_a_samples_df = pd.DataFrame({
            "context": [ex["context"] for ex in squad_answerable],
            "question": [ex["question"] for ex in squad_answerable],
            "answer": [ex["answers"]["text"][0] for ex in squad_answerable],
            "label_ID": 1,
            "label_A": 1,
            "label": 1  # ID & answerable
        })
        
        # Extract unanswerable samples (ID-U)
        squad_unanswerable = [ex for ex in squad if len(ex["answers"]["text"]) == 0]
        id_u_samples_df = pd.DataFrame({
            "context": [ex["context"] for ex in squad_unanswerable],
            "question": [ex["question"] for ex in squad_unanswerable],
            "answer": [''] * len(squad_unanswerable),
            "label_ID": 1,
            "label_A": 0,
            "label": 0  # ID & unanswerable
        })
        
        # Sample specified amounts
        id_a_samples_df = id_a_samples_df.sample(n=id_a_samples, random_state=42)
        id_u_samples_df = id_u_samples_df.sample(n=id_u_samples, random_state=43)
        
        # Construct OOD-U samples (matching ad.py logic)
        trivia_questions = list(set([
            q for q in trivia["question"] if isinstance(q, str) and q.strip() != ""
        ]))
        
        # Random sample TriviaQA questions
        ood_questions = random.sample(trivia_questions, k=ood_u_samples)
        
        # Random sample SQuAD contexts (not guaranteed to match, just maintaining structure)
        squad_contexts = random.sample([ex["context"] for ex in squad], k=ood_u_samples)
        
        # Construct OOD-U samples
        ood_u_samples_df = pd.DataFrame({
            "context": squad_contexts,
            "question": ood_questions,
            "answer": [''] * ood_u_samples,
            "label_ID": 0,
            "label_A": 0,
            "label": 0  # Non-ID, all anomalies
        })
        
        # Combine all samples
        full_data = pd.concat([id_a_samples_df, id_u_samples_df, ood_u_samples_df], ignore_index=True)
        
        # Shuffle
        full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("📊 Full dataset created:")
        print(f"label_ID distribution (ID vs OOD):")
        print(full_data["label_ID"].value_counts())
        print(f"\nlabel_A distribution (answerable vs unanswerable):")
        print(full_data["label_A"].value_counts())
        print(f"\nlabel distribution (final anomaly detection):")
        print(full_data["label"].value_counts())
        
        return full_data
    
    def create_ood_unsupervised_dataset(self, full_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create OOD unsupervised dataset (matching ad.py implementation).
        
        Args:
            full_data: Full dataset created by create_full_dataset
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        # Training set: only use ID and answerable (normal samples)
        train_df = full_data[(full_data["label"] == 1)].sample(n=3500, random_state=42)
        
        # Validation set: ID-A (normal) + OOD-U (anomaly)
        val_id = full_data[(full_data["label"] == 1)].drop(train_df.index).sample(n=1000, random_state=43)
        val_ood = full_data[(full_data["label_ID"] == 0)].sample(n=1000, random_state=44)
        val_df = pd.concat([val_id, val_ood], ignore_index=True).sample(frac=1, random_state=45)
        
        # Test set: ID-A (normal) + OOD-U (anomaly)
        test_id = full_data[(full_data["label"] == 1)].drop(train_df.index).drop(val_id.index).sample(n=1000, random_state=46)
        test_ood = full_data[(full_data["label_ID"] == 0)].drop(val_ood.index).sample(n=1500, random_state=47)
        test_df = pd.concat([test_id, test_ood], ignore_index=True).sample(frac=1, random_state=48)
        
        print("OOD Unsupervised dataset created:")
        print(f"Train label distribution:")
        print(train_df["label"].value_counts())
        print(f"\nVal label distribution:")
        print(val_df["label"].value_counts())
        print(f"\nTest label distribution:")
        print(test_df["label"].value_counts())
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    def create_ood_supervised_dataset(self, full_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create OOD supervised dataset (matching ad.py implementation).
        
        Args:
            full_data: Full dataset created by create_full_dataset
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        # ID / OOD split
        id_df = full_data[full_data["label_ID"] == 1]
        ood_df = full_data[full_data["label_ID"] == 0]
        
        # Training set
        id_train = id_df.sample(n=4000, random_state=88)
        ood_train = ood_df.sample(n=3000, random_state=88)
        train_df = pd.concat([id_train, ood_train], ignore_index=True).sample(frac=1, random_state=88)
        
        # Validation set
        id_val = id_df.drop(id_train.index).sample(n=1000, random_state=88)
        ood_val = ood_df.drop(ood_train.index).sample(n=1000, random_state=88)
        val_df = pd.concat([id_val, ood_val], ignore_index=True).sample(frac=1, random_state=88)
        
        # Test set
        id_test = id_df.drop(id_train.index).drop(id_val.index).sample(n=1000, random_state=88)
        ood_test = ood_df.drop(ood_train.index).drop(ood_val.index).sample(n=1000, random_state=88)
        test_df = pd.concat([id_test, ood_test], ignore_index=True).sample(frac=1, random_state=88)
        
        print("OOD Supervised dataset created:")
        print(f"Train label_ID distribution:")
        print(train_df["label_ID"].value_counts())
        print(f"\nVal label_ID distribution:")
        print(val_df["label_ID"].value_counts())
        print(f"\nTest label_ID distribution:")
        print(test_df["label_ID"].value_counts())
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    def create_answerability_dataset(self, full_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create answerability dataset (matching ad.py implementation).
        
        Args:
            full_data: Full dataset created by create_full_dataset
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        # Keep only ID samples (from SQuAD)
        id_data = full_data[full_data["label_ID"] == 1]
        
        # Answerable samples (positive class)
        answerable = id_data[id_data["label_A"] == 1].sample(n=8000, random_state=42)
        
        # Unanswerable samples (negative class)
        unanswerable = id_data[id_data["label_A"] == 0].sample(n=8000, random_state=43)
        
        # Training set: 5000 each
        answerable_train = answerable[:5000]
        unanswerable_train = unanswerable[:5000]
        
        # Validation set: 1500 each
        answerable_val = answerable[5000:6500]
        unanswerable_val = unanswerable[5000:6500]
        
        # Test set: 1500 each
        answerable_test = answerable[6500:]
        unanswerable_test = unanswerable[6500:]
        
        # Combine and shuffle
        train_df = pd.concat([answerable_train, unanswerable_train], ignore_index=True).sample(frac=1, random_state=100)
        val_df = pd.concat([answerable_val, unanswerable_val], ignore_index=True).sample(frac=1, random_state=101)
        test_df = pd.concat([answerable_test, unanswerable_test], ignore_index=True).sample(frac=1, random_state=102)
        
        print("✅ Answerability dataset has been created and saved:")
        print("- ans_train.csv")
        print("- ans_val.csv")
        print("- ans_test.csv")
        
        # Print sample counts
        print("=== Sample Counts ===")
        print(f"Train: {len(train_df)}")
        print(f"Val:   {len(val_df)}")
        print(f"Test:  {len(test_df)}\n")
        
        # Check label_A distribution (0: unanswerable, 1: answerable)
        print("=== Answerability Label Distribution (label_A) ===")
        print("Train:")
        print(train_df["label_A"].value_counts(), "\n")
        
        print("Validation:")
        print(val_df["label_A"].value_counts(), "\n")
        
        print("Test:")
        print(test_df["label_A"].value_counts(), "\n")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    def create_prompt_test_dataset(self, full_data: pd.DataFrame,
                                 id_a_samples: int = 1000,
                                 id_u_samples: int = 500,
                                 ood_u_samples: int = 500) -> pd.DataFrame:
        """
        Create prompt test dataset for GPT-based methods (matching ad.py).
        
        Args:
            full_data: Full dataset created by create_full_dataset
            id_a_samples: Number of ID-A samples
            id_u_samples: Number of ID-U samples
            ood_u_samples: Number of OOD-U samples
            
        Returns:
            DataFrame for prompt testing
        """
        # Construct three types of data
        id_a = full_data[(full_data["label_ID"] == 1) & (full_data["label_A"] == 1)].sample(n=id_a_samples, random_state=42)
        id_u = full_data[(full_data["label_ID"] == 1) & (full_data["label_A"] == 0)].sample(n=id_u_samples, random_state=42)
        ood_u = full_data[full_data["label_ID"] == 0].sample(n=ood_u_samples, random_state=42)
        
        # Combine and shuffle
        prompt_test_df = pd.concat([id_a, id_u, ood_u], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        print(f"✅ Prompt test dataset created with {len(prompt_test_df)} samples:")
        print(prompt_test_df["label"].value_counts().sort_index())
        
        return prompt_test_df
    
    def save_dataset(self, dataset: pd.DataFrame, filepath: str):
        """
        Save dataset to file.
        
        Args:
            dataset: DataFrame to save
            filepath: Path to save the file
        """
        dataset.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], save_dir: str):
        """
        Save multiple datasets to directory.
        
        Args:
            datasets: Dictionary with dataset names and DataFrames
            save_dir: Directory to save datasets
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, dataset in datasets.items():
            filepath = os.path.join(save_dir, f"{name}.csv")
            dataset.to_csv(filepath, index=False)
            print(f"Saved {name}.csv: {len(dataset)} samples")
    
    def load_dataset_from_file(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(filepath) 