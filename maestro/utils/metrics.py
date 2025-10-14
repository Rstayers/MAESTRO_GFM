import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # simple macro-F1 without sklearn to keep MVP minimal
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))


def label_free_utility(probs_list):
    # MVP: utility = -mean entropy across models (if committee) or single
    if not isinstance(probs_list, list):
        probs_list = [probs_list]
    ents = [entropy(p).mean() for p in probs_list]
    return -float(np.mean(ents))