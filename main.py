"""
Main script for anomaly detection in closed-domain QA systems.
Integrates all modules and executes the complete OOD detection pipeline.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json

# Add src to path
sys.path.append('src')

from data.loader import QADataLoader
from data.embeddings import SBERTExtractor, BERTExtractor, LLaMAExtractor, save_embeddings, load_embeddings
from ood.unsupervised.methods import (
    OneClassSVMDetector, GMMDetector, KDEDetector, DistanceBasedDetector, evaluate_ood_detector
)
from ood.supervised.models import MLPDetector, LogisticRegressionDetector, evaluate_supervised_detector


def create_directories():
    """Create necessary directories for outputs."""
    directories = ['data', 'embeddings', 'models', 'results', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def extract_embeddings(data_loader, embedding_type='sbert', save_path='embeddings'):
    """
    Extract embeddings from datasets.
    
    Args:
        data_loader: Data loader instance
        embedding_type: Type of embedding to extract
        save_path: Path to save embeddings
        
    Returns:
        Dictionary with embeddings and labels
    """
    print(f"Extracting {embedding_type.upper()} embeddings...")
    
    # Initialize embedding extractor
    if embedding_type == 'sbert':
        extractor = SBERTExtractor()
    elif embedding_type == 'bert_cls':
        extractor = BERTExtractor()
    elif embedding_type == 'bert_mean':
        extractor = BERTExtractor()
    elif embedding_type == 'bert_last':
        extractor = BERTExtractor()
    elif embedding_type == 'llama':
        extractor = LLaMAExtractor()
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    # Load datasets
    datasets = data_loader.create_ood_supervised_dataset()
    
    results = {}
    
    for split_name, dataset in datasets.items():
        print(f"Processing {split_name} split...")
        
        # Extract embeddings based on type
        if embedding_type == 'sbert':
            embeddings = extractor.extract_embeddings(dataset['question'].tolist())
        elif embedding_type == 'bert_cls':
            embeddings = extractor.extract_cls_embeddings(dataset['question'].tolist())
        elif embedding_type == 'bert_mean':
            embeddings = extractor.extract_mean_pooling_embeddings(dataset['question'].tolist())
        elif embedding_type == 'bert_last':
            embeddings = extractor.extract_last_token_embeddings(dataset['question'].tolist())
        elif embedding_type == 'llama':
            embeddings = extractor.extract_embeddings(dataset['question'].tolist())
        
        labels = dataset['label_ID'].values
        
        # Save embeddings
        embedding_file = os.path.join(save_path, f'{embedding_type}_{split_name}.npz')
        save_embeddings(embeddings, labels, embedding_file)
        
        results[split_name] = {
            'embeddings': embeddings,
            'labels': labels,
            'filepath': embedding_file
        }
    
    return results


def run_unsupervised_experiments(embeddings_data, results_path='results'):
    """
    Run unsupervised OOD detection experiments.
    
    Args:
        embeddings_data: Dictionary with embeddings and labels
        results_path: Path to save results
        
    Returns:
        Dictionary with results
    """
    print("Running unsupervised OOD detection experiments...")
    
    train_embeddings = embeddings_data['train']['embeddings']
    train_labels = embeddings_data['train']['labels']
    val_embeddings = embeddings_data['validation']['embeddings']
    val_labels = embeddings_data['validation']['labels']
    test_embeddings = embeddings_data['test']['embeddings']
    test_labels = embeddings_data['test']['labels']
    
    # Initialize detectors
    detectors = {
        'One-Class SVM': OneClassSVMDetector(),
        'GMM': GMMDetector(),
        'KDE': KDEDetector(),
        'Cosine Distance': DistanceBasedDetector('cosine'),
        'Euclidean Distance': DistanceBasedDetector('euclidean'),
        'Mahalanobis Distance': DistanceBasedDetector('mahalanobis')
    }
    
    results = {}
    
    for method_name, detector in detectors.items():
        print(f"\nTraining {method_name}...")
        
        try:
            # Fit detector
            if method_name in ['One-Class SVM']:
                # Use only ID data for One-Class SVM
                id_mask = train_labels == 1
                detector.fit(train_embeddings[id_mask])
            else:
                detector.fit(train_embeddings)
            
            # Predict scores
            test_scores = detector.predict_scores(test_embeddings)
            
            # Evaluate
            result = evaluate_ood_detector(test_scores, test_labels, method_name)
            results[method_name] = result
            
            # Save model
            model_file = os.path.join(results_path, f'{method_name.lower().replace(" ", "_")}.pkl')
            detector.save_model(model_file)
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            continue
    
    return results


def run_supervised_experiments(embeddings_data, results_path='results'):
    """
    Run supervised OOD detection experiments.
    
    Args:
        embeddings_data: Dictionary with embeddings and labels
        results_path: Path to save results
        
    Returns:
        Dictionary with results
    """
    print("Running supervised OOD detection experiments...")
    
    train_embeddings = embeddings_data['train']['embeddings']
    train_labels = embeddings_data['train']['labels']
    val_embeddings = embeddings_data['validation']['embeddings']
    val_labels = embeddings_data['validation']['labels']
    test_embeddings = embeddings_data['test']['embeddings']
    test_labels = embeddings_data['test']['labels']
    
    # Initialize detectors
    input_dim = train_embeddings.shape[1]
    detectors = {
        'MLP': MLPDetector(input_dim=input_dim),
        'Logistic Regression': LogisticRegressionDetector()
    }
    
    results = {}
    
    for method_name, detector in detectors.items():
        print(f"\nTraining {method_name}...")
        
        try:
            # Fit detector
            detector.fit(train_embeddings, train_labels, val_embeddings, val_labels)
            
            # Predict scores
            test_scores = detector.predict_scores(test_embeddings)
            
            # Evaluate
            result = evaluate_supervised_detector(test_scores, test_labels, method_name)
            results[method_name] = result
            
            # Save model
            model_file = os.path.join(results_path, f'{method_name.lower().replace(" ", "_")}.pkl')
            detector.save_model(model_file)
            
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            continue
    
    return results


def plot_results(unsupervised_results, supervised_results, plots_path='plots'):
    """
    Plot and save results.
    
    Args:
        unsupervised_results: Results from unsupervised experiments
        supervised_results: Results from supervised experiments
        plots_path: Path to save plots
    """
    print("Creating plots...")
    
    # Combine all results
    all_results = {**unsupervised_results, **supervised_results}
    
    if not all_results:
        print("No results to plot")
        return
    
    # Create comparison plot
    methods = list(all_results.keys())
    auc_scores = [all_results[method]['auc'] for method in methods]
    f1_scores = [all_results[method]['f1_score'] for method in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC comparison
    bars1 = ax1.bar(methods, auc_scores, color='skyblue')
    ax1.set_title('ROC-AUC Scores Comparison')
    ax1.set_ylabel('ROC-AUC Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, auc_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # F1-Score comparison
    bars2 = ax2.bar(methods, f1_scores, color='lightcoral')
    ax2.set_title('F1-Scores Comparison')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to JSON
    results_file = os.path.join(plots_path, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    print(f"Plots saved to {plots_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Anomaly Detection in Closed-Domain QA')
    parser.add_argument('--embedding_type', type=str, default='sbert',
                       choices=['sbert', 'bert_cls', 'bert_mean', 'bert_last', 'llama'],
                       help='Type of embedding to use')
    parser.add_argument('--skip_embedding', action='store_true',
                       help='Skip embedding extraction and use existing files')
    parser.add_argument('--unsupervised_only', action='store_true',
                       help='Run only unsupervised experiments')
    parser.add_argument('--supervised_only', action='store_true',
                       help='Run only supervised experiments')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Initialize data loader
    data_loader = QADataLoader(random_seed=42)
    
    # Extract or load embeddings
    if args.skip_embedding:
        print("Loading existing embeddings...")
        embeddings_data = {}
        for split in ['train', 'validation', 'test']:
            embedding_file = f'embeddings/{args.embedding_type}_{split}.npz'
            if os.path.exists(embedding_file):
                embeddings, labels = load_embeddings(embedding_file)
                embeddings_data[split] = {
                    'embeddings': embeddings,
                    'labels': labels,
                    'filepath': embedding_file
                }
            else:
                raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    else:
        embeddings_data = extract_embeddings(data_loader, args.embedding_type)
    
    # Run experiments
    unsupervised_results = {}
    supervised_results = {}
    
    if not args.supervised_only:
        unsupervised_results = run_unsupervised_experiments(embeddings_data)
    
    if not args.unsupervised_only:
        supervised_results = run_supervised_experiments(embeddings_data)
    
    # Plot and save results
    plot_results(unsupervised_results, supervised_results)
    
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main() 