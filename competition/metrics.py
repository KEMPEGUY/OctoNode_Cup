# competition/metrics.py

from sklearn.metrics import f1_score
import numpy as np


def macro_f1(y_true, y_pred):
    """
    Compute macro F1 score.
    Ensures integer class labels.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)

    return float(f1_score(y_true, y_pred, average="macro"))
