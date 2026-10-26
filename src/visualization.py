"""
src/visualization.py
Generazione di tutti i grafici del progetto FOODRISK.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR, CLASS_NAMES, PALETTE, FIGURE_DPI
from src.ontology_features import OWL_FEATURE_NAMES

os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_metrics_barplot(aggregated: dict, model_names: list):
    """
    Barplot di F1 / Precision / Recall / G-Mean per modello.
    aggregated: {model_name: {metric_mean: val, metric_std: val, ...}}
    """
    metrics = ["f1_macro", "precision", "recall", "g_mean"]
    labels  = ["F1 macro", "Precision", "Recall", "G-Mean"]
    x = np.arange(len(model_names))
    width = 0.20

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (met, lbl) in enumerate(zip(metrics, labels)):
        means = [aggregated[m][f"{met}_mean"] for m in model_names]
        stds  = [aggregated[m][f"{met}_std"]  for m in model_names]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
               label=lbl, color=PALETTE[i], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("FOODRISK — Metriche per modello (mean ± std, 10 run)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "metrics_barplot.png")
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()
    print(f"[viz] Salvato: {path}")


def plot_f1_boxplot(f1_per_run: dict):
    """
    Boxplot di F1 macro su 10 run per ogni modello.
    f1_per_run: {model_name: list of f1 (len=n_runs)}
    """
    model_names = list(f1_per_run.keys())
    data = [f1_per_run[m] for m in model_names]

    fig, ax = plt.subplots(figsize=(13, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE * 3):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 macro")
    ax.set_title("FOODRISK — Distribuzione F1 macro (10 run)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "f1_boxplot.png")
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()
    print(f"[viz] Salvato: {path}")


def plot_nemenyi_heatmap(pmat: np.ndarray, model_names: list):
    """
    Heatmap p-value post-hoc Nemenyi.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(model_names), dtype=bool)
    sns.heatmap(pmat, annot=True, fmt=".3f", cmap="YlOrRd_r",
                xticklabels=model_names, yticklabels=model_names,
                mask=mask, ax=ax, vmin=0, vmax=1,
                linewidths=0.5, linecolor="white")
    ax.set_title("Post-hoc Nemenyi — p-value\n(rosso = differenza significativa)")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "nemenyi_heatmap.png")
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()
    print(f"[viz] Salvato: {path}")


def plot_owl_coverage(coverage: np.ndarray):
    """
    Heatmap (n_classes × n_bk_concepts): attivazione media dei concetti OWL
    per classe. Mostra quali concetti BK sono più informativi per ogni profilo.
    """
    short_names = [n.replace("bk_", "") for n in OWL_FEATURE_NAMES]
    class_labels = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(coverage, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=short_names, yticklabels=class_labels,
                linewidths=0.4, linecolor="white", ax=ax, vmin=0, vmax=1)
    ax.set_title("Copertura concetti OWL per profilo\n(attivazione media su training set)")
    ax.set_xlabel("Concetto BK")
    ax.set_ylabel("Profilo")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "owl_coverage.png")
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()
    print(f"[viz] Salvato: {path}")


def plot_em_convergence(ll_history: list, run: int = 0):
    """Log-likelihood dell'EM nel tempo (per un run)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ll_history, marker="o", color=PALETTE[0])
    ax.set_xlabel("Iterazione EM")
    ax.set_ylabel("Log-likelihood media (dati etichettati)")
    ax.set_title(f"Convergenza BNB-EM — run {run}")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"em_convergence_run{run}.png")
    plt.savefig(path, dpi=FIGURE_DPI)
    plt.close()
    print(f"[viz] Salvato: {path}")
