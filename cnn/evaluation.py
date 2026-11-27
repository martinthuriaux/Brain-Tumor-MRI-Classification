'''
Evaluates the best CNN model saved by train.py on the test set.

train.py keeps only the best epoch of each run (highest validation accuracy), saved as
checkpoints/cnn<N>_epoch<E>_best_model.pth. This script evaluates the latest run (highest N).

We try to follow the instructions from Python for Data Science, Lecture 7

The epoch was already chosen on the validation set during training, so the test set is used only
once, here, and the test score is never used to make a choice.

Run from anywhere in the repo:
    python cnn/evaluation.py

Figures and CSV files are saved to results/cnn<N>/.
'''

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from data_preprocessing import get_loaders
from model import CNN

# Set root so that the script can run from anywhere in the repo
ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"

# Colors for the plots, chosen to be colorblind friendly :)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# Decide what device to use: GPU if available, otherwise CPU. This is the same logic as in train.py.
def get_device():
    """Use an NVIDIA GPU or an Apple Silicon GPU if there is one, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# From the checkpoint folder, find the latest run's best model (highest N in cnn<N>_epoch<E>_best_model.pth).
def find_best_model(folder):
    """Return (run, epoch, path) of the latest run's best model: the cnn<N>_epoch<E>_best_model.pth with the highest N."""
    found = []
    for path in folder.glob("cnn*_best_model.pth"):
        match = re.fullmatch(r"cnn(\d+)_epoch(\d+)_best_model\.pth", path.name)
        if match:
            found.append((int(match.group(1)), int(match.group(2)), path))
    if not found:
        raise SystemExit(f"No cnn<N>_epoch<E>_best_model.pth files found in {folder}. Run cnn/train.py first.")
    # tuples compare by their first item, so max() picks the highest run number
    return max(found)

# function we use to rebuild the CNN and load the weights saved by train.py. 
def load_model(path, num_classes, device):
    """Rebuild the CNN and load the weights saved by train.py."""
    model = CNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model

# function we use to run the model over every batch of the loader and return the predicted probabilities, true labels, and average cross-entropy loss.
@torch.no_grad()   # we only predict here, so PyTorch does not need to keep track of gradients
def predict(model, loader, device):
    """Run the model over every batch of the loader.

    Returns the predicted probabilities (one row per image, one column per class),
    the true labels, and the average cross-entropy loss (the same loss train.py prints).
    """
    all_probs, all_labels = [], []
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        # softmax turns the four raw scores into probabilities that add up to 1
        all_probs.append(logits.softmax(dim=1).cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return probs, labels, total_loss / len(labels)

def plot_confusion_matrix(cm, class_names, model_label, out_dir):
    """Heatmap of the confusion matrix (Lecture 7: sns.heatmap with annot=True).

    Rows are the true classes and columns the predicted ones, so the diagonal counts the correct
    predictions and every other cell is one kind of mistake. Counts only go up from 0, so we use a
    single-hue color scale (light = few images, dark = many) instead of the diverging "coolwarm"
    used for correlations, which run from -1 to +1.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", vmin=0,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1.5, linecolor="white",
                cbar_kws={"label": "Number of test images"}, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix, test set ({model_label})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)


def plot_per_class_metrics(per_class, model_label, out_dir):
    """Precision, recall and F1 for each class (Lecture 7: a bar plot to compare categories).

    precision: of the images predicted as class X, the share that really are X.
    recall:    of the images that really are X, the share the model found.
    f1-score:  the harmonic mean of the two, high only if both are high.
    """
    # seaborn expects "long" data: one row per (class, metric) pair
    long = (per_class[["precision", "recall", "f1-score"]]
            .rename_axis("class").reset_index()
            .melt(id_vars="class", var_name="metric", value_name="score"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=long, x="class", y="score", hue="metric",
                palette=[BLUE, ORANGE, AQUA], saturation=1, gap=0.15, ax=ax)
    ax.set_ylim(0, 1)   # bars must start at 0, otherwise small differences look huge
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class metrics, test set ({model_label})")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_metrics.png", dpi=150)

def main():
    sns.set_theme(style="whitegrid")
    device = get_device()
    
    # get information about the best model from the checkpoint directory
    run, epoch, model_path = find_best_model(CHECKPOINT_DIR)
    model_label = f"cnn{run}, epoch {epoch}"

    # one results folder per run, so evaluating a new run never overwrites an earlier one
    out_dir = ROOT / "results" / f"cnn{run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # same split, same class order and same (non-augmented) preprocessing as train.py
    _, val_loader, test_loader, class_names = get_loaders()
    val_subset = val_loader.dataset
    val_targets = [val_subset.dataset.targets[i] for i in val_subset.indices]
    counts = pd.DataFrame({
        "validation": pd.Series([class_names[t] for t in val_targets]).value_counts(),
        "test": pd.Series([class_names[t] for t in test_loader.dataset.targets]).value_counts(),
    }).loc[class_names]

    # Print some information about the model and the dataset
    print(f"Device: {device}")
    print(f"Model: {model_path.relative_to(ROOT)}")
    print(f"\nImages per class:\n{counts}")

    
    # Score the best model once on the test set
    model = load_model(model_path, len(class_names), device)
    probs, labels, test_loss = predict(model, test_loader, device)
    # The predicted class is the one with the highest probability for each image
    preds = probs.argmax(axis=1)

    # one row per test image; test_loader does not shuffle, so the rows follow dataset.samples
    predictions = pd.DataFrame({
        "path": [str(Path(path).relative_to(ROOT)) for path, _ in test_loader.dataset.samples],
        "true": [class_names[i] for i in labels],
        "predicted": [class_names[i] for i in preds],
        "confidence": probs.max(axis=1),
        "outcome": np.where(labels == preds, "correct", "misclassified"),
    })

    
    for i, name in enumerate(class_names):
        predictions[f"prob_{name}"] = probs[:, i]
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    accuracy = accuracy_score(labels, preds)
    # macro F1: the F1-score of each class separately, then averaged, so each class counts equally
    macro_f1 = f1_score(labels, preds, average="macro")

    precision, recall, f1_per_class, support = precision_recall_fscore_support(labels, preds)
    per_class = pd.DataFrame({"precision": precision, "recall": recall,
                              "f1-score": f1_per_class, "support": support}, index=class_names)
    per_class.round(4).to_csv(out_dir / "test_per_class_metrics.csv", index_label="class")

    pd.DataFrame([{
        "model": model_path.name, "run": run, "epoch": epoch, "test_loss": test_loss,
        "accuracy": accuracy, "macro_f1": macro_f1,
    }]).round(4).to_csv(out_dir / "test_summary.csv", index=False)

    cm = confusion_matrix(labels, preds)

    print(f"\nPer-class metrics (test set, {model_label}):\n{per_class.round(3)}")
    # center (mean, median = 50%) and spread (std, quartiles) of the confidence, as in Lecture 7
    print(f"\nConfidence of the predictions (test set):\n"
          f"{predictions.groupby('outcome')['confidence'].describe().round(3)}")

    
    plot_confusion_matrix(cm, class_names, model_label, out_dir)
    plot_per_class_metrics(per_class, model_label, out_dir)

    
    print(f"\nSummary (test set, {model_path.name})")
    print(f"  Accuracy: {100 * accuracy:.1f}%")
    print(f"  Macro F1: {macro_f1:.3f}")

    mistakes_only = cm.copy()
    np.fill_diagonal(mistakes_only, 0)
    if mistakes_only.any():
        true_i, pred_i = np.unravel_index(mistakes_only.argmax(), cm.shape)
        print(f"  Most frequent mistake: {cm[true_i, pred_i]} of {cm[true_i].sum()} "
              f"{class_names[true_i]} images predicted as {class_names[pred_i]}")
    median_conf = predictions.groupby("outcome")["confidence"].median()
    for outcome, value in median_conf.items():
        print(f"  Median confidence when {outcome}: {value:.2f}")

    print(f"\nFigures and CSV files saved to {out_dir.relative_to(ROOT)}/")
    plt.show()


if __name__ == "__main__":
    main()
