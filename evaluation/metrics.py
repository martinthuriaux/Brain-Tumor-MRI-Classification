
import numpy as np
import torch

"""
MODEL EVALUATION - CLASSIFICATION METRICS

This module evaluates the performance of our brain tumor
MRI classification models.

It contains two main functions:

1. compute_metrics():
   Calculates accuracy, precision, recall, F1-score,
   and the confusion matrix from true and predicted labels.

2. evaluate_model():
   Runs a PyTorch model on an evaluation dataset,
   collects its predictions, and computes the metrics.

The code works with our custom CNN and can also be
used with transfer learning models such as ResNet.

Macro-averaged metrics give equal importance to each
class, which is useful when class sizes are different.

The results will be used to compare the performance
of different models on the same test dataset.
"""





from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, num_classes=4):
    """
    Compute classification metrics from true and predicted labels.
    """
    labels = list(range(num_classes))

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ),
    }


def evaluate_model(model, dataloader, device="cpu", num_classes=4):
    """
    Evaluate a CNN or transfer learning classification model.

    Returns metrics and predictions for further analysis.
    """
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if not all_labels:
        raise ValueError("The evaluation dataloader is empty.")

    results = compute_metrics(
        all_labels,
        all_predictions,
        num_classes=num_classes,
    )

    results["y_true"] = np.array(all_labels)
    results["y_pred"] = np.array(all_predictions)

    return results
