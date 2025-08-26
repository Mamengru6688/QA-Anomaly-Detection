
# Anomaly Detection Framework for Closed-Domain Question Answering

This repository implements a flexible and modular framework for **anomaly detection** in **closed-domain question answering (QA)** systems. The goal is to improve **LLM reliability** by detecting questions that are either **out-of-domain (OOD)** or **unanswerable** based on the given context—thus reducing hallucinated or misleading answers.

We explore and compare two complementary detection strategies:

1. **Modular Pipeline**: 
   - Separates OOD detection and answerability detection into dedicated components.
   - Uses pretrained embeddings (e.g., SBERT, LLaMA) with lightweight models (e.g., SVM, MLP).

2. **Prompt-Based Detection**:
   - Reformulates anomaly detection as a single-step inference task via instruction tuning.
   - Employs prompting techniques like zero-shot, few-shot, and chain-of-thought (CoT) reasoning.


## Project Structure

```
project_root/
├── README.md
├── main.py
├── requirements.txt
└── src/
    ├── data/             # Data loading, processing, and utilities
    ├── ood/
    │   ├── unsupervised/ # Unsupervised OOD methods
    │   └── supervised/   # Supervised OOD methods
    ├── answerability/    # Answerability detection module
    └── prompt/           # Prompt-based detection methods
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

The project uses the SQuAD 2.0 dataset and TriviaQA.

## Evaluation

All methods are evaluated using:
- ROC-AUC scores
- F1-score
