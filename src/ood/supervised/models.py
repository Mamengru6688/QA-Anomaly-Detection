"""
Supervised OOD detection models.
Includes Multi-layer Perceptron and Logistic Regression.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
from typing import Tuple, Dict, Optional


class QADataset(Dataset):
    """Dataset for QA embeddings."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            embeddings: Input embeddings
            labels: Target labels
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """Multi-layer Perceptron for OOD detection."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256, 128], num_classes: int = 2):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
        """
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class SupervisedOODDetector:
    """Base class for supervised OOD detection."""
    
    def __init__(self, method_name: str):
        """
        Initialize the detector.
        
        Args:
            method_name: Name of the detection method
        """
        self.method_name = method_name
        self.model = None
        self.is_fitted = False
    
    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray,
            val_embeddings: Optional[np.ndarray] = None, 
            val_labels: Optional[np.ndarray] = None):
        """
        Fit the model on training data.
        
        Args:
            train_embeddings: Training embeddings
            train_labels: Training labels
            val_embeddings: Validation embeddings
            val_labels: Validation labels
        """
        raise NotImplementedError
    
    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        raise NotImplementedError
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the model file
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


class MLPDetector(SupervisedOODDetector):
    """Multi-layer Perceptron for OOD detection."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256, 128], 
                 learning_rate: float = 0.001, batch_size: int = 32, 
                 num_epochs: int = 50, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize MLP detector.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of training epochs
            device: Device to run the model on
        """
        super().__init__('MLP')
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.model = MLPClassifier(input_dim, hidden_dims).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray,
            val_embeddings: Optional[np.ndarray] = None, 
            val_labels: Optional[np.ndarray] = None):
        """
        Fit MLP on training data.
        
        Args:
            train_embeddings: Training embeddings
            train_labels: Training labels
            val_embeddings: Validation embeddings
            val_labels: Validation labels
        """
        # Create datasets
        train_dataset = QADataset(train_embeddings, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if val_embeddings is not None and val_labels is not None:
            val_dataset = QADataset(val_embeddings, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_embeddings, batch_labels in train_loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_embeddings)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_embeddings, batch_labels in val_loader:
                        batch_embeddings = batch_embeddings.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = self.model(batch_embeddings)
                        loss = self.criterion(outputs, batch_labels)
                        val_loss += loss.item()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                          f"Val Loss = {val_loss/len(val_loader):.4f}")
        
        self.is_fitted = True
        print(f"MLP fitted on {len(train_embeddings)} samples")
    
    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using softmax probabilities.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        dataset = QADataset(embeddings, np.zeros(len(embeddings)))  # Dummy labels
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        scores = []
        
        with torch.no_grad():
            for batch_embeddings, _ in dataloader:
                batch_embeddings = batch_embeddings.to(self.device)
                outputs = self.model(batch_embeddings)
                probs = torch.softmax(outputs, dim=1)
                # Use probability of OOD class (class 0) as anomaly score
                ood_probs = probs[:, 0].cpu().numpy()
                scores.extend(ood_probs)
        
        return np.array(scores)


class LogisticRegressionDetector(SupervisedOODDetector):
    """Logistic Regression for OOD detection."""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """
        Initialize Logistic Regression detector.
        
        Args:
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
        """
        super().__init__('Logistic Regression')
        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self, train_embeddings: np.ndarray, train_labels: np.ndarray,
            val_embeddings: Optional[np.ndarray] = None, 
            val_labels: Optional[np.ndarray] = None):
        """
        Fit Logistic Regression on training data.
        
        Args:
            train_embeddings: Training embeddings
            train_labels: Training labels
            val_embeddings: Validation embeddings (not used for LR)
            val_labels: Validation labels (not used for LR)
        """
        # Scale features
        train_embeddings_scaled = self.scaler.fit_transform(train_embeddings)
        
        # Fit model
        self.model.fit(train_embeddings_scaled, train_labels)
        self.is_fitted = True
        print(f"Logistic Regression fitted on {len(train_embeddings)} samples")
    
    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using probability of OOD class.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Get probabilities
        probs = self.model.predict_proba(embeddings_scaled)
        # Use probability of OOD class (class 0) as anomaly score
        ood_probs = probs[:, 0]
        
        return ood_probs


def evaluate_supervised_detector(scores: np.ndarray, 
                               labels: np.ndarray, 
                               method_name: str) -> Dict[str, float]:
    """
    Evaluate supervised OOD detector performance.
    
    Args:
        scores: Anomaly scores
        labels: True labels (0 for OOD, 1 for ID)
        method_name: Name of the method
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert scores to predictions
    # Higher scores = more anomalous, so we use scores directly for thresholding
    thresholds = np.percentile(scores, np.arange(0, 100, 1))
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        preds = (scores > threshold).astype(int)
        f1 = f1_score(labels, preds, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Final predictions with best threshold
    final_preds = (scores > best_threshold).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(labels, scores)
    report = classification_report(labels, final_preds, 
                                 target_names=['OOD', 'ID'], 
                                 output_dict=True)
    
    results = {
        'method': method_name,
        'auc': auc,
        'f1_score': best_f1,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'threshold': best_threshold
    }
    
    print(f"=== {method_name} Results ===")
    print(f"AUC: {auc:.4f}")
    print(f"F1-Score: {best_f1:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print()
    
    return results 