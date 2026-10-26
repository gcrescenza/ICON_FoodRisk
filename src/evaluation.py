"""
src/evaluation.py
Calcolo metriche, test di Friedman e post-hoc Nemenyi.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import friedmanchisquare


def g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Geometric Mean of per-class recall.
    G-mean = (prod_c recall_c) ^ (1/n_classes)
    Penalizza fortemente le classi ignorate dal classificatore.
    """
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recall_c = (y_pred[mask] == c).sum() / mask.sum()
        recalls.append(recall_c)
    if not recalls:
        return 0.0
    return float(np.prod(recalls) ** (1.0 / len(recalls)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Restituisce le metriche principali per un singolo run."""
    return {
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "g_mean":    g_mean(y_true, y_pred),
    }


def aggregate_runs(run_metrics: list) -> dict:
    """
    Dato una lista di dict di metriche (uno per run), restituisce
    mean ± std per ciascuna metrica.
    """
    result = {}
    keys = run_metrics[0].keys()
    for k in keys:
        vals = np.array([m[k] for m in run_metrics])
        result[k + "_mean"] = float(vals.mean())
        result[k + "_std"]  = float(vals.std())
    return result


def friedman_test(f1_matrix: np.ndarray):
    """
    Test di Friedman su matrice (n_models × n_runs).
    Restituisce (statistica chi², p-value).
    """
    # scipy vuole (n_runs × n_models) come argomenti separati
    rows = [f1_matrix[i] for i in range(f1_matrix.shape[0])]
    stat, pval = friedmanchisquare(*rows)
    return float(stat), float(pval)


def nemenyi_pvalue_matrix(f1_matrix: np.ndarray) -> np.ndarray:
    """
    Matrice dei p-value del post-hoc di Nemenyi dopo il test di Friedman.

    f1_matrix: (n_models x n_runs)
    Restituisce: matrice simmetrica (n_models x n_models), diagonale = 1.

    Se e' installato `scikit-posthocs` usa posthoc_nemenyi_friedman; altrimenti
    usa un'implementazione equivalente basata su scipy: differenza dei ranghi
    medi (rango 1 = miglior F1 nel run) standardizzata con
    sqrt(k(k+1)/(6n)) e confrontata con la distribuzione del range
    studentizzato (infiniti gradi di liberta'), come in Demsar (2006).
    """
    try:
        from scikit_posthocs import posthoc_nemenyi_friedman
        import pandas as pd
        df = pd.DataFrame(f1_matrix.T)   # (n_runs, n_models)
        return posthoc_nemenyi_friedman(df).values
    except ImportError:
        pass

    from scipy.stats import rankdata, studentized_range
    k, n = f1_matrix.shape
    ranks = np.apply_along_axis(lambda col: rankdata(-col), 0, f1_matrix)
    mean_ranks = ranks.mean(axis=1)
    diff = np.abs(mean_ranks[:, None] - mean_ranks[None, :])
    q = diff / np.sqrt(k * (k + 1) / (6.0 * n)) * np.sqrt(2.0)
    pmat = studentized_range.sf(q, k, np.inf)
    np.fill_diagonal(pmat, 1.0)
    return pmat
