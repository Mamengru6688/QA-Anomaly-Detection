"""
Answerability Detection module based on original ad.py implementation.
Handles the detection of whether a question can be answered from a given context.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import joblib
import json
import os
from tqdm import tqdm


class AnswerabilityDataProcessor:
    """Data processor for answerability detection based on original implementation."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data processor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def create_answerability_dataset(self, full_data: pd.DataFrame,
                                   answerable_samples: int = 8000,
                                   unanswerable_samples: int = 8000) -> Dict[str, pd.DataFrame]:
        """
        Create answerability dataset with train/val/test splits.
        Based on original ad.py implementation.
        
        Args:
            full_data: Full dataset with label_ID and label_A columns
            answerable_samples: Number of answerable samples to use
            unanswerable_samples: Number of unanswerable samples to use
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        # Keep only ID samples (from SQuAD)
        id_data = full_data[full_data["label_ID"] == 1]
        
        # Answerable samples (positive class)
        answerable = id_data[id_data["label_A"] == 1].sample(n=answerable_samples, random_state=42)
        
        # Unanswerable samples (negative class)
        unanswerable = id_data[id_data["label_A"] == 0].sample(n=unanswerable_samples, random_state=43)
        
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
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], save_dir: str):
        """
        Save answerability datasets.
        
        Args:
            datasets: Dictionary with train, validation, and test DataFrames
            save_dir: Directory to save the datasets
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for split_name, dataset in datasets.items():
            # Save full dataset
            dataset.to_csv(f"{save_dir}/ans_{split_name}.csv", index=False)
            
            # Save labels separately
            dataset['label_A'].to_csv(f"{save_dir}/ans_{split_name}_labels.csv", index=False)
            
            print(f"Saved {split_name} dataset: {len(dataset)} samples")
            print(f"  Answerable: {sum(dataset['label_A'] == 1)}")
            print(f"  Unanswerable: {sum(dataset['label_A'] == 0)}")
    
    def load_datasets(self, save_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load answerability datasets.
        
        Args:
            save_dir: Directory containing the datasets
            
        Returns:
            Dictionary with train, validation, and test DataFrames
        """
        datasets = {}
        
        for split_name in ['train', 'validation', 'test']:
            filepath = f"{save_dir}/ans_{split_name}.csv"
            if os.path.exists(filepath):
                datasets[split_name] = pd.read_csv(filepath)
            else:
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        return datasets


class FeatureDataset(Dataset):
    """PyTorch dataset for answerability detection features."""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPDropout(nn.Module):
    """MLP model with dropout for answerability detection."""
    
    def __init__(self, input_dim):
        super(MLPDropout, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class AnswerabilityTrainer:
    """Trainer for answerability detection models."""
    
    def __init__(self, device: str = None):
        """
        Initialize the trainer.
        
        Args:
            device: Device to use for training
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
    
    def train_mlp(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  save_dir: str = "models",
                  num_epochs: int = 50,
                  patience: int = 15) -> Dict[str, float]:
        """
        Train MLP model for answerability detection.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            save_dir: Directory to save model
            num_epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Dictionary with test results
        """
        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = FeatureDataset(X_train_scaled, y_train)
        val_dataset = FeatureDataset(X_val_scaled, y_val)
        test_dataset = FeatureDataset(X_test_scaled, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Create model
        self.model = MLPDropout(X_train.shape[1]).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        
        # Training process (with Early Stopping)
        best_val_auc = 0.0
        patience_counter = 0
        train_losses, val_losses, val_aucs = [], [], []
        
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            all_outputs, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).unsqueeze(1)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    all_outputs.extend(outputs.sigmoid().cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            val_auc = roc_auc_score(all_labels, all_outputs)
            val_aucs.append(val_auc)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), f"{save_dir}/best_mlp_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        
        # Final test evaluation (using best model)
        self.model.load_state_dict(torch.load(f"{save_dir}/best_mlp_model.pt"))
        self.model.eval()
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).unsqueeze(1)
                outputs = self.model(X_batch)
                all_outputs.extend(outputs.sigmoid().cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        test_probs = np.array(all_outputs).flatten()
        test_preds = (test_probs >= 0.5).astype(int)
        test_auc = roc_auc_score(all_labels, test_probs)
        test_report = classification_report(all_labels, test_preds, target_names=["Unanswerable", "Answerable"], digits=4)
        
        # Save ROC curve data
        fpr, tpr, _ = roc_curve(all_labels, test_probs)
        roc_data = {
            "method": "pytorch_mlp",
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
        
        with open(f"{save_dir}/roc_pytorch_mlp.json", "w") as f:
            json.dump(roc_data, f)
        
        print("=== Test ===")
        print(test_report)
        print(f"AUC: {test_auc:.4f}")
        
        return {
            'auc': test_auc,
            'f1_score': f1_score(all_labels, test_preds, average='weighted'),
            'precision': test_report['weighted avg']['precision'],
            'recall': test_report['weighted avg']['recall']
        }
    
    def train_sklearn_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            save_dir: str = "models") -> Dict[str, Dict[str, float]]:
        """
        Train sklearn models for answerability detection.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            save_dir: Directory to save models
            
        Returns:
            Dictionary with results from all models
        """
        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to try
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "lightgbm": LGBMClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        os.makedirs(save_dir, exist_ok=True)
        
        # Train and evaluate each model
        for name, clf in models.items():
            print(f"\n=== Training {name} ===")
            clf.fit(X_train_scaled, y_train)

            # Validation set evaluation
            val_probs = clf.predict_proba(X_val_scaled)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
            print(f"Val AUC: {val_auc:.4f}")

            # Test set evaluation
            test_probs = clf.predict_proba(X_test_scaled)[:, 1]
            test_preds = (test_probs >= 0.5).astype(int)
            test_auc = roc_auc_score(y_test, test_probs)
            test_report = classification_report(y_test, test_preds, target_names=["Unanswerable", "Answerable"], digits=4)
            print(f"Test AUC: {test_auc:.4f}")
            print(test_report)

            # Save model
            joblib.dump(clf, f"{save_dir}/{name}_model.pkl")

            # Save ROC curve data
            fpr, tpr, _ = roc_curve(y_test, test_probs)
            roc_data = {
                "method": name,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
            with open(f"{save_dir}/roc_{name}.json", "w") as f:
                json.dump(roc_data, f)
            
            results[name] = {
                'auc': test_auc,
                'f1_score': f1_score(y_test, test_preds, average='weighted'),
                'precision': test_report['weighted avg']['precision'],
                'recall': test_report['weighted avg']['recall']
            }
        
        return results


def plot_answerability_results(results: Dict[str, Dict[str, float]], 
                             save_path: str = 'plots/answerability_results.png'):
    """
    Plot answerability detection results.
    
    Args:
        results: Dictionary with results from different methods
        save_path: Path to save the plot
    """
    if not results:
        print("No results to plot")
        return
    
    methods = list(results.keys())
    auc_scores = [results[method]['auc'] for method in methods]
    f1_scores = [results[method]['f1_score'] for method in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC comparison
    bars1 = ax1.bar(methods, auc_scores, color='lightblue')
    ax1.set_title('Answerability Detection - ROC-AUC Scores')
    ax1.set_ylabel('ROC-AUC Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, auc_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # F1-Score comparison
    bars2 = ax2.bar(methods, f1_scores, color='lightgreen')
    ax2.set_title('Answerability Detection - F1-Scores')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {save_path}") 