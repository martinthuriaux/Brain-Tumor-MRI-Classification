
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay





"""
EVALUATION VISUALIZATIONS

This module creates visualizations of model performance.

It contains two main functions:

1. plot_confusion_matrix():
   Shows how many images were correctly or incorrectly
   classified for each class.

   Rows represent true labels.
   Columns represent predicted labels.

2. plot_model_comparison():
   Creates a bar chart comparing accuracy, precision,
   recall, and F1-score across different models.

The figures can be saved as PNG files in the
figures/ directory.

These visualizations help us identify classification
errors and compare the CNN with transfer learning models.
"""


def plot_confusion_matrix(
    cm,
    class_names,
    save_path=None,
):
    """
    Plot a confusion matrix using class names.
    """
    cm = np.asarray(cm)

    if cm.shape != (len(class_names), len(class_names)):
        raise ValueError("Matrix size must match class names.")

    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(ax=ax, cmap="Blues", colorbar=True)

    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_comparison(results, save_path=None):
    """
    Compare accuracy, precision, recall and F1-score
    across multiple models.

    results = {
        "CNN": metrics_dict,
        "ResNet": metrics_dict,
    }
    """
    metric_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    model_names = list(results.keys())

    if not model_names:
        raise ValueError("No models provided.")

    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(model_names):
        values = [results[name][key] for key in metric_keys]

        ax.bar(
            x + (i - (len(model_names) - 1) / 2) * width,
            values,
            width,
            label=name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
