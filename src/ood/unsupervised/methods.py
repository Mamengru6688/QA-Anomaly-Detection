"""
Unsupervised OOD detection methods.
Includes One-Class SVM, Gaussian Mixture Models, and Kernel Density Estimation.
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import joblib
import json
from typing import Tuple, List, Dict, Optional


class UnsupervisedOODDetector:
    """Base class for unsupervised OOD detection."""
    
    def __init__(self, method_name: str):
        """
        Initialize the detector.
        
        Args:
            method_name: Name of the detection method
        """
        self.method_name = method_name
        self.model = None
        self.is_fitted = False
    
    def fit(self, train_embeddings: np.ndarray):
        """
        Fit the model on training data.
        
        Args:
            train_embeddings: Training embeddings
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


class OneClassSVMDetector(UnsupervisedOODDetector):
    """One-Class SVM for OOD detection."""
    
    def __init__(self, kernel: str = 'rbf', nu: float = 0.1, gamma: str = 'scale'):
        """
        Initialize One-Class SVM detector.
        
        Args:
            kernel: SVM kernel type
            nu: An upper bound on the fraction of training errors
            gamma: Kernel coefficient
        """
        super().__init__('One-Class SVM')
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    
    def fit(self, train_embeddings: np.ndarray):
        """
        Fit One-Class SVM on training data.
        
        Args:
            train_embeddings: Training embeddings (only in-distribution)
        """
        self.model.fit(train_embeddings)
        self.is_fitted = True
        print(f"One-Class SVM fitted on {len(train_embeddings)} samples")
    
    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using decision function.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Decision function returns distance to the separating hyperplane
        # Negative values indicate inliers, positive values indicate outliers
        scores = -self.model.decision_function(embeddings)
        return scores


class GMMDetector(UnsupervisedOODDetector):
    """Gaussian Mixture Model for OOD detection."""
    
    def __init__(self, n_components: int = 2, covariance_type: str = 'full', random_state: int = 42):
        """
        Initialize GMM detector.
        
        Args:
            n_components: Number of mixture components
            covariance_type: Type of covariance parameters
            random_state: Random seed
        """
        super().__init__('GMM')
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = GaussianMixture(n_components=n_components, 
                                    covariance_type=covariance_type, 
                                    random_state=random_state)
    
    def fit(self, train_embeddings: np.ndarray):
        """
        Fit GMM on training data.
        
        Args:
            train_embeddings: Training embeddings
        """
        self.model.fit(train_embeddings)
        self.is_fitted = True
        print(f"GMM fitted on {len(train_embeddings)} samples with {self.n_components} components")
    
    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using log-likelihood.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Log-likelihood scores (higher = more likely to be in-distribution)
        scores = self.model.score_samples(embeddings)
        return scores


class KDEDetector(UnsupervisedOODDetector):
    """Kernel Density Estimation for OOD detection."""
    
    def __init__(self, kernel: str = 'gaussian', bandwidth: float = 1.0):
        """
        Initialize KDE detector.
        
        Args:
            kernel: Kernel type
            bandwidth: Bandwidth parameter
        """
        super().__init__('KDE')
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    
    def fit(self, train_embeddings: np.ndarray):
        """
        Fit KDE on training data.
        
        Args:
            train_embeddings: Training embeddings
        """
        self.model.fit(train_embeddings)
        self.is_fitted = True
        print(f"KDE fitted on {len(train_embeddings)} samples")
    
    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using log-likelihood.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Log-likelihood scores (higher = more likely to be in-distribution)
        scores = self.model.score_samples(embeddings)
        return scores


class DistanceBasedDetector:
    """Distance-based OOD detection methods."""
    
    def __init__(self, method: str = 'cosine'):
        """
        Initialize distance-based detector.
        
        Args:
            method: Distance method ('cosine', 'euclidean', 'mahalanobis')
        """
        self.method = method
        self.reference_embeddings = None
        self.covariance_matrix = None
    
    def fit(self, train_embeddings: np.ndarray):
        """
        Fit the distance-based detector.
        
        Args:
            train_embeddings: Training embeddings
        """
        self.reference_embeddings = train_embeddings
        
        if self.method == 'mahalanobis':
            # Compute covariance matrix for Mahalanobis distance
            self.covariance_matrix = np.cov(train_embeddings.T)
        
        print(f"Distance-based detector fitted on {len(train_embeddings)} samples")
    
    def predict_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using distance measures.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if self.reference_embeddings is None:
            raise ValueError("Model must be fitted before prediction")
        
        if self.method == 'cosine':
            # Cosine similarity (higher = more similar, so we negate for anomaly scores)
            similarities = cosine_similarity(embeddings, self.reference_embeddings)
            scores = -np.max(similarities, axis=1)
        
        elif self.method == 'euclidean':
            # Euclidean distance (higher = more distant)
            distances = euclidean_distances(embeddings, self.reference_embeddings)
            scores = np.min(distances, axis=1)
        
        elif self.method == 'mahalanobis':
            # Mahalanobis distance
            if self.covariance_matrix is None:
                raise ValueError("Covariance matrix not computed")
            
            scores = []
            for emb in embeddings:
                distances = [mahalanobis(emb, ref_emb, self.covariance_matrix) 
                           for ref_emb in self.reference_embeddings]
                scores.append(np.min(distances))
            scores = np.array(scores)
        
        else:
            raise ValueError(f"Unknown distance method: {self.method}")
        
        return scores


def evaluate_ood_detector(scores: np.ndarray, 
                         labels: np.ndarray, 
                         method_name: str,
                         reverse: bool = True) -> Dict[str, float]:
    """
    Evaluate OOD detector performance.
    
    Args:
        scores: Anomaly scores
        labels: True labels (0 for OOD, 1 for ID)
        method_name: Name of the method
        reverse: Whether to reverse the scores for thresholding
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert scores to predictions
    if reverse:
        # Higher scores = more anomalous, so reverse for thresholding
        threshold_scores = -scores
    else:
        threshold_scores = scores
    
    # Find optimal threshold
    thresholds = np.percentile(threshold_scores, np.arange(0, 100, 1))
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        preds = (threshold_scores > threshold).astype(int)
        f1 = f1_score(labels, preds, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Final predictions with best threshold
    final_preds = (threshold_scores > best_threshold).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(labels, threshold_scores)
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