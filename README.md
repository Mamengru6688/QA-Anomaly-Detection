# Anomaly Detection in Closed-Domain QA

This repository contains a comprehensive implementation of anomaly detection (out-of-distribution detection) methods for closed-domain question answering systems.

## Project Structure

```
project_root/
├── README.md
├── main.py
├── requirements.txt
└── src/
    ├── data/             # Data loading, processing, and utilities
    ├── ood/
    │   ├── unsupervised/ # Unsupervised OOD methods (SVM, GMM, KDE)
    │   └── supervised/   # Supervised OOD methods (MLP, LR)
    ├── answerability/    # Answerability detection module
    └── prompt/           # Prompt-based detection methods
```

## Features

- **Unsupervised OOD Detection**: One-Class SVM, Gaussian Mixture Models (GMM), Kernel Density Estimation (KDE)
- **Supervised OOD Detection**: Multi-layer Perceptron (MLP), Logistic Regression
- **Distance-based Methods**: Cosine similarity, Euclidean distance, Mahalanobis distance
- **Embedding-based Methods**: BERT, SBERT, LLaMA embeddings
- **Prompt-based Methods**: OpenAI GPT-based detection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Data

The project uses the SQuAD dataset for in-distribution samples and TriviaQA for out-of-distribution samples.

## Methods

### Unsupervised Methods
- **One-Class SVM**: Trained only on in-distribution data
- **GMM**: Gaussian Mixture Models for density estimation
- **KDE**: Kernel Density Estimation for non-parametric density modeling

### Supervised Methods
- **MLP**: Multi-layer Perceptron with BERT embeddings
- **Logistic Regression**: Linear classifier with BERT embeddings

### Distance-based Methods
- **Cosine Similarity**: Based on embedding similarity
- **Euclidean Distance**: L2 distance between embeddings
- **Mahalanobis Distance**: Distance considering covariance structure

## Evaluation

All methods are evaluated using:
- ROC-AUC scores
- Classification reports (precision, recall, F1-score)
- ROC curves visualization 