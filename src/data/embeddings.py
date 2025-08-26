"""
Embedding extraction module for anomaly detection.
Handles BERT, SBERT, and LLaMA embeddings.
"""

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, LlamaTokenizer, LlamaModel
from tqdm import tqdm
import os
from typing import List, Tuple, Optional


class EmbeddingExtractor:
    """Base class for embedding extraction."""
    
    def __init__(self, model_name: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the embedding extractor.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings from texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        raise NotImplementedError


class SBERTExtractor(EmbeddingExtractor):
    """SBERT embedding extractor."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize SBERT extractor.
        
        Args:
            model_name: SBERT model name
            device: Device to run the model on
        """
        super().__init__(model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract SBERT embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting SBERT embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


class BERTExtractor(EmbeddingExtractor):
    """BERT embedding extractor."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize BERT extractor.
        
        Args:
            model_name: BERT model name
            device: Device to run the model on
        """
        super().__init__(model_name, device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def extract_cls_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract CLS token embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of CLS embeddings
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT CLS embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch_texts, 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512, 
                                  return_tensors='pt').to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    def extract_mean_pooling_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract mean pooling embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of mean pooling embeddings
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT mean pooling embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch_texts, 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512, 
                                  return_tensors='pt').to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                
                # Mean pooling
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                mean_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = mean_embeddings.cpu().numpy()
                embeddings.append(mean_embeddings)
        
        return np.vstack(embeddings)
    
    def extract_last_token_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract last token embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of last token embeddings
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT last token embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch_texts, 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512, 
                                  return_tensors='pt').to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                
                # Last token pooling
                token_embeddings = outputs.last_hidden_state
                last_token_positions = attention_mask.sum(dim=1) - 1
                last_token_embeddings = token_embeddings[torch.arange(token_embeddings.size(0)), last_token_positions]
                last_token_embeddings = last_token_embeddings.cpu().numpy()
                embeddings.append(last_token_embeddings)
        
        return np.vstack(embeddings)


class LLaMAExtractor(EmbeddingExtractor):
    """LLaMA embedding extractor."""
    
    def __init__(self, model_name: str = 'meta-llama/Llama-2-7b-hf', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize LLaMA extractor.
        
        Args:
            model_name: LLaMA model name
            device: Device to run the model on
        """
        super().__init__(model_name, device)
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            self.model = LlamaModel.from_pretrained(model_name).to(device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load LLaMA model {model_name}: {e}")
            print("Please ensure you have access to the LLaMA model or use a different model.")
            self.tokenizer = None
            self.model = None
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Extract LLaMA embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing (smaller for LLaMA)
            
        Returns:
            Array of embeddings
        """
        if self.model is None:
            raise ValueError("LLaMA model not loaded. Please check model access.")
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting LLaMA embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch_texts, 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512, 
                                  return_tensors='pt').to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state mean pooling
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                mean_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = mean_embeddings.cpu().numpy()
                embeddings.append(mean_embeddings)
        
        return np.vstack(embeddings)


def save_embeddings(embeddings: np.ndarray, labels: np.ndarray, filepath: str):
    """
    Save embeddings and labels to file.
    
    Args:
        embeddings: Embedding array
        labels: Label array
        filepath: Path to save the file
    """
    np.savez(filepath, embeddings=embeddings, labels=labels)
    print(f"Embeddings saved to {filepath}")


def load_embeddings(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and labels from file.
    
    Args:
        filepath: Path to the embeddings file
        
    Returns:
        Tuple of (embeddings, labels)
    """
    data = np.load(filepath)
    return data['embeddings'], data['labels'] 