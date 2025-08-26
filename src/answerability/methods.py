"""
Answerability detection feature extraction methods based on original ad.py implementation.
Includes BERT and LLaMA feature extraction with various pooling strategies.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import os


class BERTFeatureExtractor:
    """BERT feature extractor for answerability detection."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize BERT feature extractor.
        
        Args:
            model_name: BERT model name
        """
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
    
    def create_inputs(self, df: pd.DataFrame, order: str = "qc") -> List[str]:
        """
        Create inputs for BERT.
        
        Args:
            df: DataFrame with question and context columns
            order: Input order ('qc' for question+context, 'cq' for context+question)
            
        Returns:
            List of formatted input strings
        """
        if order == "qc":
            return (df["question"] + " [SEP] " + df["context"]).tolist()
        elif order == "cq":
            return (df["context"] + " [SEP] " + df["question"]).tolist()
        else:
            raise ValueError("order must be 'qc' or 'cq'")
    
    def extract_mean_pooling_features(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract mean pooling features from BERT.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Extracted features
        """
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT mean pooling features"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state  # shape: (B, T, D)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)

                # Masked mean pooling
                summed = torch.sum(token_embeddings * attention_mask, dim=1)
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = (summed / counts).cpu().numpy()
                all_embeddings.append(mean_pooled)
        
        return np.vstack(all_embeddings)
    
    def extract_cls_features(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract CLS token features from BERT.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Extracted features
        """
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT CLS features"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
        
        return np.vstack(all_embeddings)
    
    def extract_last_token_features(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract last token features from BERT.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Extracted features
        """
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT last token features"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_token_embeddings = []
                for j in range(outputs.last_hidden_state.size(0)):
                    seq_len = inputs["attention_mask"][j].sum()
                    last_token = outputs.last_hidden_state[j, seq_len - 1, :]  # Last actual token
                    last_token_embeddings.append(last_token.cpu().numpy())
                all_embeddings.extend(last_token_embeddings)
        
        return np.vstack(all_embeddings)
    
    def extract_combined_features(self, questions: List[str], contexts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Extract combined features: CLS + Q-pooling + C-pooling + interaction.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            batch_size: Batch size for processing
            
        Returns:
            Extracted features
        """
        all_features = []
        
        # Load model with hidden states output
        model = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        with torch.no_grad():
            for i in tqdm(range(0, len(questions), batch_size), desc="Extracting BERT combined features"):
                batch_questions = questions[i:i+batch_size]
                batch_contexts = contexts[i:i+batch_size]
                batch_texts = [c + " [SEP] " + q for q, c in zip(batch_questions, batch_contexts)]  # C + Q

                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}

                outputs = model(**inputs)
                last_hidden = outputs.hidden_states[-1]  # Last layer hidden states (B, T, H)

                for j in range(last_hidden.size(0)):
                    input_ids = inputs["input_ids"][j]
                    attn_mask = inputs["attention_mask"][j]

                    # === CLS token ===
                    cls_token = last_hidden[j, 0, :]  # (H)

                    # === SEP position identification ===
                    sep_pos = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=False).flatten()
                    if len(sep_pos) >= 2:
                        c_start, c_end = 1, sep_pos[0].item()
                        q_start, q_end = sep_pos[0].item() + 1, sep_pos[1].item()
                    else:
                        # Fallback: simple split
                        seq_len = attn_mask.sum().item()
                        c_start, c_end = 1, seq_len // 2
                        q_start, q_end = seq_len // 2, seq_len

                    # === Mean pooling ===
                    question_repr = last_hidden[j, q_start:q_end, :].mean(dim=0)
                    context_repr = last_hidden[j, c_start:c_end, :].mean(dim=0)

                    # === Interaction term (element-wise multiplication) ===
                    interaction = question_repr * context_repr

                    # === Concatenate into one vector ===
                    combined = torch.cat([cls_token, question_repr, context_repr, interaction], dim=0)  # (4H)
                    all_features.append(combined.cpu())

        return torch.stack(all_features).numpy()
    
    def extract_cls_interaction_features(self, questions: List[str], contexts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Extract CLS + interaction features.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            batch_size: Batch size for processing
            
        Returns:
            Extracted features
        """
        all_features = []
        
        # Load model with hidden states output
        model = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        with torch.no_grad():
            for i in tqdm(range(0, len(questions), batch_size), desc="Extracting BERT CLS+interaction features"):
                batch_questions = questions[i:i+batch_size]
                batch_contexts = contexts[i:i+batch_size]
                batch_texts = [c + " [SEP] " + q for q, c in zip(batch_questions, batch_contexts)]  # C + Q

                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}

                outputs = model(**inputs)
                last_hidden = outputs.hidden_states[-1]  # Last layer hidden states (B, T, H)

                for j in range(last_hidden.size(0)):
                    input_ids = inputs["input_ids"][j]
                    attn_mask = inputs["attention_mask"][j]

                    # === CLS token ===
                    cls_token = last_hidden[j, 0, :]  # (H)

                    # === SEP position identification ===
                    sep_pos = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=False).flatten()
                    if len(sep_pos) >= 2:
                        c_start, c_end = 1, sep_pos[0].item()
                        q_start, q_end = sep_pos[0].item() + 1, sep_pos[1].item()
                    else:
                        seq_len = attn_mask.sum().item()
                        c_start, c_end = 1, seq_len // 2
                        q_start, q_end = seq_len // 2, seq_len

                    # Mean pooling
                    question_repr = last_hidden[j, q_start:q_end, :].mean(dim=0)
                    context_repr = last_hidden[j, c_start:c_end, :].mean(dim=0)

                    # Interaction
                    interaction = question_repr * context_repr

                    # Concatenate CLS + interaction
                    combined = torch.cat([cls_token, interaction], dim=0)  # (2H)
                    all_features.append(combined.cpu())

        return torch.stack(all_features).numpy()


class LLaMAFeatureExtractor:
    """LLaMA feature extractor for answerability detection."""
    
    def __init__(self, model_name: str = "NousResearch/Llama-2-7b-chat-hf"):
        """
        Initialize LLaMA feature extractor.
        
        Args:
            model_name: LLaMA model name
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
    
    def extract_last_token_features(self, questions: List[str], contexts: List[str], 
                                  order: str = "cq", batch_size: int = 4) -> np.ndarray:
        """
        Extract last token features from LLaMA.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            order: Input order ('cq' for context+question, 'qc' for question+context)
            batch_size: Batch size for processing
            
        Returns:
            Extracted features
        """
        features = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Extracting LLaMA last token features ({order})"):
            batch_q = questions[i:i+batch_size]
            batch_c = contexts[i:i+batch_size]
            
            if order == "cq":
                inputs = [c + " " + self.tokenizer.eos_token + " " + q for q, c in zip(batch_q, batch_c)]
            else:  # qc
                inputs = [q + " " + self.tokenizer.eos_token + " " + c for q, c in zip(batch_q, batch_c)]

            encodings = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**encodings, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer (batch, seq_len, hidden_dim)

            # Extract the last non-padding token vector for each sample
            input_ids = encodings['input_ids']
            for j in range(input_ids.size(0)):
                attention_mask = encodings['attention_mask'][j]
                last_idx = attention_mask.nonzero()[-1].item()
                last_token_embedding = hidden_states[j, last_idx, :].cpu().numpy()
                features.append(last_token_embedding)

        return np.array(features)
    
    def extract_mean_pooling_features(self, questions: List[str], contexts: List[str], 
                                    order: str = "cq", batch_size: int = 4) -> np.ndarray:
        """
        Extract mean pooling features from LLaMA.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            order: Input order ('cq' for context+question, 'qc' for question+context)
            batch_size: Batch size for processing
            
        Returns:
            Extracted features
        """
        all_features = []
        
        # Mean pooling function
        def mean_pooling(hidden_states, attention_mask):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            return sum_embeddings / sum_mask
        
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Extracting LLaMA mean pooling features ({order})"):
            batch_questions = questions[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            
            if order == "cq":
                batch_texts = [c + " " + self.tokenizer.eos_token + " " + q for q, c in zip(batch_questions, batch_contexts)]
            else:  # qc
                batch_texts = [q + " " + self.tokenizer.eos_token + " " + c for q, c in zip(batch_questions, batch_contexts)]

            encodings = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**encodings, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                pooled = mean_pooling(last_hidden, encodings["attention_mask"])
                all_features.append(pooled.cpu())

        return torch.cat(all_features, dim=0).numpy()
    
    def extract_first_token_features(self, questions: List[str], contexts: List[str], 
                                   order: str = "cq", batch_size: int = 4) -> np.ndarray:
        """
        Extract first token features from LLaMA.
        
        Args:
            questions: List of questions
            contexts: List of contexts
            order: Input order ('cq' for context+question, 'qc' for question+context)
            batch_size: Batch size for processing
            
        Returns:
            Extracted features
        """
        all_features = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Extracting LLaMA first token features ({order})"):
            batch_questions = questions[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size]
            
            if order == "cq":
                batch_texts = [c + " [SEP] " + q for q, c in zip(batch_questions, batch_contexts)]
            else:  # qc
                batch_texts = [q + " [SEP] " + c for q, c in zip(batch_questions, batch_contexts)]

            encodings = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**encodings, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                first_token = last_hidden[:, 0, :]  # (batch_size, hidden_dim)
                all_features.append(first_token.cpu())

        return torch.cat(all_features, dim=0).numpy()


def extract_and_save_features(datasets: Dict[str, pd.DataFrame], 
                            extractor_type: str = "bert",
                            feature_type: str = "mean_pooling",
                            input_order: str = "cq",
                            save_dir: str = "embeddings") -> Dict[str, np.ndarray]:
    """
    Extract and save features for answerability detection.
    
    Args:
        datasets: Dictionary with train, validation, and test DataFrames
        extractor_type: Type of extractor ('bert' or 'llama')
        feature_type: Type of features to extract
        input_order: Input order ('cq' or 'qc')
        save_dir: Directory to save features
        
    Returns:
        Dictionary with extracted features
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if extractor_type == "bert":
        extractor = BERTFeatureExtractor()
        
        # Create inputs
        train_texts = extractor.create_inputs(datasets['train'], input_order)
        val_texts = extractor.create_inputs(datasets['validation'], input_order)
        test_texts = extractor.create_inputs(datasets['test'], input_order)
        
        # Extract features
        if feature_type == "mean_pooling":
            train_features = extractor.extract_mean_pooling_features(train_texts)
            val_features = extractor.extract_mean_pooling_features(val_texts)
            test_features = extractor.extract_mean_pooling_features(test_texts)
            suffix = "mean"
        elif feature_type == "cls":
            train_features = extractor.extract_cls_features(train_texts)
            val_features = extractor.extract_cls_features(val_texts)
            test_features = extractor.extract_cls_features(test_texts)
            suffix = "cls"
        elif feature_type == "last_token":
            train_features = extractor.extract_last_token_features(train_texts)
            val_features = extractor.extract_last_token_features(val_texts)
            test_features = extractor.extract_last_token_features(test_texts)
            suffix = "lasttoken"
        elif feature_type == "combined":
            train_features = extractor.extract_combined_features(
                datasets['train']["question"].tolist(), 
                datasets['train']["context"].tolist()
            )
            val_features = extractor.extract_combined_features(
                datasets['validation']["question"].tolist(), 
                datasets['validation']["context"].tolist()
            )
            test_features = extractor.extract_combined_features(
                datasets['test']["question"].tolist(), 
                datasets['test']["context"].tolist()
            )
            suffix = "combo"
        elif feature_type == "cls_interaction":
            train_features = extractor.extract_cls_interaction_features(
                datasets['train']["question"].tolist(), 
                datasets['train']["context"].tolist()
            )
            val_features = extractor.extract_cls_interaction_features(
                datasets['validation']["question"].tolist(), 
                datasets['validation']["context"].tolist()
            )
            test_features = extractor.extract_cls_interaction_features(
                datasets['test']["question"].tolist(), 
                datasets['test']["context"].tolist()
            )
            suffix = "cls_inter"
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    elif extractor_type == "llama":
        extractor = LLaMAFeatureExtractor()
        
        questions_train = datasets['train']["question"].tolist()
        contexts_train = datasets['train']["context"].tolist()
        questions_val = datasets['validation']["question"].tolist()
        contexts_val = datasets['validation']["context"].tolist()
        questions_test = datasets['test']["question"].tolist()
        contexts_test = datasets['test']["context"].tolist()
        
        if feature_type == "last_token":
            train_features = extractor.extract_last_token_features(questions_train, contexts_train, input_order)
            val_features = extractor.extract_last_token_features(questions_val, contexts_val, input_order)
            test_features = extractor.extract_last_token_features(questions_test, contexts_test, input_order)
            suffix = "last"
        elif feature_type == "mean_pooling":
            train_features = extractor.extract_mean_pooling_features(questions_train, contexts_train, input_order)
            val_features = extractor.extract_mean_pooling_features(questions_val, contexts_val, input_order)
            test_features = extractor.extract_mean_pooling_features(questions_test, contexts_test, input_order)
            suffix = "mean"
        elif feature_type == "first_token":
            train_features = extractor.extract_first_token_features(questions_train, contexts_train, input_order)
            val_features = extractor.extract_first_token_features(questions_val, contexts_val, input_order)
            test_features = extractor.extract_first_token_features(questions_test, contexts_test, input_order)
            suffix = "first"
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    # Save features
    np.save(f"{save_dir}/ans_train_{extractor_type}_{suffix}_{input_order}.npy", train_features)
    np.save(f"{save_dir}/ans_val_{extractor_type}_{suffix}_{input_order}.npy", val_features)
    np.save(f"{save_dir}/ans_test_{extractor_type}_{suffix}_{input_order}.npy", test_features)
    
    # Save labels
    datasets['train']["label_A"].to_csv(f"{save_dir}/ans_train_labels.csv", index=False)
    datasets['validation']["label_A"].to_csv(f"{save_dir}/ans_val_labels.csv", index=False)
    datasets['test']["label_A"].to_csv(f"{save_dir}/ans_test_labels.csv", index=False)
    
    print(f"✅ {extractor_type.upper()} {feature_type} features ({input_order}) extracted and saved.")
    
    return {
        'train': train_features,
        'validation': val_features,
        'test': test_features
    } 