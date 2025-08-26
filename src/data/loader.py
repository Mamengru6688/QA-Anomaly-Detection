"""
Data loader module for anomaly detection in QA systems.
Handles loading and preprocessing of SQuAD and TriviaQA datasets.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import random
from typing import Tuple, Dict, List


class QADataLoader:
    """Data loader for question-answering datasets."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data loader.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def load_squad_data(self, split: str = 'train') -> pd.DataFrame:
        """
        Load SQuAD dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            
        Returns:
            DataFrame with questions and contexts
        """
        dataset = load_dataset('squad', split=split)
        
        # Extract questions and contexts
        data = []
        for item in dataset:
            data.append({
                'question': item['question'],
                'context': item['context'],
                'answer': item['answers']['text'][0] if item['answers']['text'] else '',
                'label_ID': 1  # In-distribution
            })
        
        return pd.DataFrame(data)
    
    def load_trivia_data(self, split: str = 'train') -> pd.DataFrame:
        """
        Load TriviaQA dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            
        Returns:
            DataFrame with questions
        """
        dataset = load_dataset('trivia_qa', 'rc.nocontext', split=split)
        
        # Extract questions
        data = []
        for item in dataset:
            data.append({
                'question': item['question'],
                'context': '',  # TriviaQA doesn't have contexts
                'answer': item['answer']['value'],
                'label_ID': 0  # Out-of-distribution
            })
        
        return pd.DataFrame(data)
    
    def create_ood_unsupervised_dataset(self, 
                                       squad_samples: int = 5000,
                                       trivia_samples: int = 5000) -> pd.DataFrame:
        """
        Create OOD unsupervised dataset.
        
        Args:
            squad_samples: Number of SQuAD samples to use
            trivia_samples: Number of TriviaQA samples to use
            
        Returns:
            Combined dataset with ID and OOD samples
        """
        # Load SQuAD data (in-distribution)
        squad_df = self.load_squad_data('train')
        squad_df = squad_df.sample(n=squad_samples, random_state=self.random_seed)
        
        # Load TriviaQA data (out-of-distribution)
        trivia_df = self.load_trivia_data('train')
        trivia_df = trivia_df.sample(n=trivia_samples, random_state=self.random_seed)
        
        # Combine datasets
        full_data = pd.concat([squad_df, trivia_df], ignore_index=True)
        
        return full_data
    
    def create_ood_supervised_dataset(self,
                                     id_train_samples: int = 3000,
                                     id_val_samples: int = 1000,
                                     id_test_samples: int = 1000,
                                     ood_train_samples: int = 3000,
                                     ood_val_samples: int = 1000,
                                     ood_test_samples: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Create OOD supervised dataset with train/val/test splits.
        
        Args:
            id_train_samples: Number of ID training samples
            id_val_samples: Number of ID validation samples
            id_test_samples: Number of ID test samples
            ood_train_samples: Number of OOD training samples
            ood_val_samples: Number of OOD validation samples
            ood_test_samples: Number of OOD test samples
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        # Load SQuAD data (ID)
        squad_df = self.load_squad_data('train')
        
        # Load TriviaQA data (OOD)
        trivia_df = self.load_trivia_data('train')
        
        # Split ID data
        id_train = squad_df.sample(n=id_train_samples, random_state=self.random_seed)
        remaining_id = squad_df.drop(id_train.index)
        id_val = remaining_id.sample(n=id_val_samples, random_state=self.random_seed)
        remaining_id = remaining_id.drop(id_val.index)
        id_test = remaining_id.sample(n=id_test_samples, random_state=self.random_seed)
        
        # Split OOD data
        ood_train = trivia_df.sample(n=ood_train_samples, random_state=self.random_seed)
        remaining_ood = trivia_df.drop(ood_train.index)
        ood_val = remaining_ood.sample(n=ood_val_samples, random_state=self.random_seed)
        remaining_ood = remaining_ood.drop(ood_val.index)
        ood_test = remaining_ood.sample(n=ood_test_samples, random_state=self.random_seed)
        
        # Combine and shuffle
        train_df = pd.concat([id_train, ood_train], ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        val_df = pd.concat([id_val, ood_val], ignore_index=True)
        val_df = val_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        test_df = pd.concat([id_test, ood_test], ignore_index=True)
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    def save_dataset(self, dataset: pd.DataFrame, filepath: str):
        """
        Save dataset to file.
        
        Args:
            dataset: DataFrame to save
            filepath: Path to save the file
        """
        dataset.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset_from_file(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(filepath) 