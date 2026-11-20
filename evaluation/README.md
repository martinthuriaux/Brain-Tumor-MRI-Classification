

# Model Evaluation and Comparison

## Table of Contents

1. [Objective](#1-objective)
2. [Common Experimental Protocol](#2-common-experimental-protocol)
3. [Required Outputs from Each Model](#3-required-outputs-from-each-model)
   - [Model checkpoint](#model-checkpoint)
   - [Training history](#training-history)
   - [Experimental configuration](#experimental-configuration)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Comparative Analysis](#5-comparative-analysis)
6. [Limitations](#6-limitations)
7. [Evaluation Module Structure](#7-evaluation-module-structure)

---




## 1. Objective

The objective of this module is to evaluate and compare the CNN 
trained from scratch and the pretrained model using a consistent 
and reproducible methodology.

## 2. Common Experimental Protocol

To ensure a fair comparison, both models must:

- Use identical training, validation and test splits.
- Use consistent class-to-index mappings.
- Use the same evaluation metrics.
- Use deterministic validation and test transformations.
- Select the best model using validation performance.
- Keep the test set separate from hyperparameter tuning.
- Document model-specific preprocessing.

The current dataset splitting configuration is:

- Validation fraction: 15%
- Split random seed: 42
- Image resolution: 224 x 224

## 3. Required Outputs from Each Model

### Model checkpoint

- Saved weights (.pth)
- Model architecture
- Class-to-index mapping

### Training history

For each epoch, record:

- Epoch number
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Epoch duration in seconds

Preferably, save these results in a CSV file.

### Experimental configuration

- Optimizer
- Learning rate
- Batch size
- Number of epochs
- Random seed
- Loss function
- Image preprocessing and augmentation
- Hardware used (CPU/GPU)
- Total training time
- Number of total and trainable parameters

## 4. Evaluation Metrics

Both models will be evaluated using:

- Accuracy
- Precision per class
- Recall per class
- F1-score per class
- Macro F1-score
- Confusion matrix

## 5. Comparative Analysis

The comparison will include:

- Overall classification performance
- Performance by tumor category
- Training and validation learning curves
- Convergence speed
- Computational efficiency
- Misclassified images
- Statistical significance of performance differences
- Confidence intervals

## 6. Limitations

- Potential duplicate images across dataset splits.
- Absence of patient identifiers.
- Potential correlations between MRI images.
- Limited generalization to unseen patients and hospitals.

The results must not be interpreted as evidence of clinical 
diagnostic reliability.

## 7. Evaluation Module Structure

| File | Purpose |
|---|---|
| metrics.py | Classification metrics |
| plots.py | Performance visualizations |
| statistical_tests.py | Statistical comparison |
| error_analysis.py | Analysis of misclassified images |
| comparison.ipynb | Final comparative analysis |
