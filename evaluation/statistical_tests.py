
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar




"""
STATISTICAL COMPARISON OF CLASSIFICATION MODELS

This module compares the predictions of two models
using McNemar's statistical test.

McNemar's test is designed for paired observations:
both models must predict the same test images.

The test focuses on cases where the models disagree
about whether a prediction is correct.

The function compare_models_mcnemar() returns:

- A 2x2 contingency table
- The test statistic
- The p-value
- Whether the result is significant at alpha = 0.05

A p-value below 0.05 provides evidence against
the null hypothesis of equal error probabilities
for the two models.

This allows us to investigate whether differences
between models are statistically significant,
rather than comparing accuracy alone.
"""


def compare_models_mcnemar(
    y_true,
    predictions_a,
    predictions_b,
):
    """
    Compare two classifiers evaluated on the same samples.

    Uses the exact McNemar test.
    """
    y_true = np.asarray(y_true)
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)

    if (
        y_true.ndim != 1
        or predictions_a.shape != y_true.shape
        or predictions_b.shape != y_true.shape
    ):
        raise ValueError(
            "All inputs must be 1D arrays of equal length."
        )

    if len(y_true) == 0:
        raise ValueError("Predictions cannot be empty.")

    correct_a = predictions_a == y_true
    correct_b = predictions_b == y_true

    both_correct = np.sum(correct_a & correct_b)
    only_a_correct = np.sum(correct_a & ~correct_b)
    only_b_correct = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)

    table = np.array([
        [both_correct, only_a_correct],
        [only_b_correct, both_wrong],
    ])

    result = mcnemar(table, exact=True)

    return {
        "contingency_table": table,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_0_05": bool(result.pvalue < 0.05),
    }
