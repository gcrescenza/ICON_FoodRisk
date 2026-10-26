"""
main.py — Pipeline principale FOODRISK
Esegue tutti i modelli su N_RUNS run di StratifiedShuffleSplit,
raccoglie le metriche e genera i grafici.
"""

import os
import sys
import numpy as np
import pandas as pd

from config import (N_RUNS, RANDOM_SEED, CLASS_NAMES, N_CLASSES,
                    RESULTS_DIR, DATASET_PATH)

# ── Generazione dataset se non esiste ────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    print("[main] Dataset non trovato — generazione automatica...")
    import data.generate_dataset as gen
    gen.main()

# ── Costruzione ontologia se non esiste ──────────────────────────────────────
from config import ONTOLOGY_PATH
if not os.path.exists(ONTOLOGY_PATH):
    print("[main] Ontologia non trovata — costruzione automatica...")
    from ontology.build_ontology import build_ontology
    build_ontology()

# ── Coerenza ontologia <-> KB Prolog <-> fallback Python ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb"))
from kb.verify_prolog import check_static
_problems = check_static()
if _problems:
    raise SystemExit("[main] KB Prolog non coerente con ontologia/fallback:\n  " + "\n  ".join(_problems))
print("[main] Verifica statica OK: ontologia, food_kb.pl e fallback Python coerenti.")

# ── Import moduli ─────────────────────────────────────────────────────────────
from src.preprocessing import load_data, get_splits
from src.ontology_features import (extract_owl_features, augment_with_owl,
                                    owl_coverage_per_class)
from src.supervised import get_models, tuned_fit
from src.bernoulli_nb import BernoulliNB
from src.bnb_em import BernoulliNB_EM, split_labeled_unlabeled
from src.bayesian_network import FoodBayesianNetwork
from src.prolog_kb import PrologKB
from src.evaluation import compute_metrics, aggregate_runs, friedman_test, nemenyi_pvalue_matrix
from src.visualization import (plot_metrics_barplot, plot_f1_boxplot,
                                plot_nemenyi_heatmap, plot_owl_coverage,
                                plot_em_convergence)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("  FOODRISK — Meal Nutritional Risk Classification")
    print("=" * 60)

    # ── Caricamento dati ──────────────────────────────────────────────────────
    df = load_data()
    splits, _ = get_splits(df, n_splits=N_RUNS, random_state=RANDOM_SEED)

    # ── Raccoglitori metriche ─────────────────────────────────────────────────
    # Struttura: {model_label: [metriche_run_0, metriche_run_1, ...]}
    all_metrics = {
        "DT_raw":    [], "RF_raw":    [], "LR_raw":    [], "SVM_raw":  [],
        "DT_owl":    [], "RF_owl":    [], "LR_owl":    [], "SVM_owl":  [],
        "BNB_owl":   [], "BNB_EM":    [],
        "BayesNet":  [], "KB_Prolog": [],
    }
    f1_per_run = {k: [] for k in all_metrics}

    # ── Coverage OWL (calcolata sul primo split) ──────────────────────────────
    X_tr0, _, y_tr0, _ = splits[0]
    coverage = owl_coverage_per_class(X_tr0, y_tr0, N_CLASSES)
    plot_owl_coverage(coverage)

    # ── KB Prolog ─────────────────────────────────────────────────────────────
    prolog_kb = PrologKB()

    # ── Loop sui run ─────────────────────────────────────────────────────────
    em_ll_saved = False
    kb_runs = []     # (y_true, y_pred) della KB per ogni run (metriche per classe, media ± std)
    selected = []                 # iperparametri scelti per run (CV interna)

    for run_idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
        print(f"\n── Run {run_idx + 1}/{N_RUNS} ──────────────────────────────")

        # Feature OWL
        X_train_owl = extract_owl_features(X_train)
        X_test_owl  = extract_owl_features(X_test)
        X_train_aug = augment_with_owl(X_train)
        X_test_aug  = augment_with_owl(X_test)

        # Feature OWL binarie (0/1) per BNB, BNB-EM e Rete Bayesiana
        X_train_bin = (X_train_owl >= 0.5).astype(float)
        X_test_bin  = (X_test_owl  >= 0.5).astype(float)

        # ── Modelli supervisionati ── RAW e +OWL ──────────────────────────────
        for tag, feat_tr, feat_te in [
            ("_raw", X_train, X_test),
            ("_owl", X_train_aug, X_test_aug),
        ]:
            suffix = tag
            for name, model in get_models().items():
                key = f"{name.replace('LogisticRegression','LR').replace('_RBF','').replace('DecisionTree','DT').replace('RandomForest','RF')}{suffix}"
                if key not in all_metrics:
                    continue
                model, best = tuned_fit(name, feat_tr, y_train, seed=RANDOM_SEED + run_idx)
                selected.append({"run": run_idx, "features": tag[1:], "modello": name,
                                 "parametro": list(best)[0],
                                 "valore": "None" if list(best.values())[0] is None else list(best.values())[0]})
                y_pred = model.predict(feat_te)
                m = compute_metrics(y_test, y_pred)
                all_metrics[key].append(m)
                f1_per_run[key].append(m["f1_macro"])
                print(f"  {key:20s}  F1={m['f1_macro']:.3f}  G-Mean={m['g_mean']:.3f}")

        # ── BNB (feature OWL binarie) ─────────────────────────────────────────
        bnb = BernoulliNB()
        bnb.fit(X_train_bin, y_train)
        y_pred_bnb = bnb.predict(X_test_bin)
        m_bnb = compute_metrics(y_test, y_pred_bnb)
        all_metrics["BNB_owl"].append(m_bnb)
        f1_per_run["BNB_owl"].append(m_bnb["f1_macro"])
        print(f"  {'BNB_owl':20s}  F1={m_bnb['f1_macro']:.3f}  G-Mean={m_bnb['g_mean']:.3f}")

        # ── BNB-EM (semi-supervisionato) ──────────────────────────────────────
        X_lab, y_lab, X_unlab = split_labeled_unlabeled(
            X_train_bin, y_train, seed=RANDOM_SEED + run_idx)
        bnb_em = BernoulliNB_EM()
        bnb_em.fit(X_lab, y_lab, X_unlab)
        y_pred_em = bnb_em.predict(X_test_bin)
        m_em = compute_metrics(y_test, y_pred_em)
        all_metrics["BNB_EM"].append(m_em)
        f1_per_run["BNB_EM"].append(m_em["f1_macro"])
        print(f"  {'BNB_EM':20s}  F1={m_em['f1_macro']:.3f}  G-Mean={m_em['g_mean']:.3f}  "
              f"(EM iters={bnb_em.n_iter_})")

        if not em_ll_saved and bnb_em.ll_history_:
            plot_em_convergence(bnb_em.ll_history_, run=run_idx)
            em_ll_saved = True

        # ── Rete Bayesiana (feature BK binarie, inferenza esatta) ─────────────
        bn = FoodBayesianNetwork()
        bn.fit(X_train_bin, y_train)
        y_pred_bn = bn.predict(X_test_bin)
        m_bn = compute_metrics(y_test, y_pred_bn)
        all_metrics["BayesNet"].append(m_bn)
        f1_per_run["BayesNet"].append(m_bn["f1_macro"])
        print(f"  {'BayesNet':20s}  F1={m_bn['f1_macro']:.3f}  G-Mean={m_bn['g_mean']:.3f}")

        # ── KB Prolog (ragionamento simbolico, nessun apprendimento) ──────────
        y_pred_kb = prolog_kb.infer_profile(X_test)
        m_kb = compute_metrics(y_test, y_pred_kb)
        all_metrics["KB_Prolog"].append(m_kb)
        kb_runs.append((np.asarray(y_test), np.asarray(y_pred_kb)))
        f1_per_run["KB_Prolog"].append(m_kb["f1_macro"])
        print(f"  {'KB_Prolog':20s}  F1={m_kb['f1_macro']:.3f}  G-Mean={m_kb['g_mean']:.3f}"
              + ("" if prolog_kb.available else "  [fallback Python]"))

    # ── Aggregazione risultati ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RISULTATI AGGREGATI (mean ± std su 10 run)")
    print("=" * 60)

    aggregated = {}
    rows = []
    for key, metrics_list in all_metrics.items():
        agg = aggregate_runs(metrics_list)
        aggregated[key] = agg
        rows.append({
            "Modello":    key,
            "F1 macro":   f"{agg['f1_macro_mean']:.3f} ± {agg['f1_macro_std']:.3f}",
            "Precision":  f"{agg['precision_mean']:.3f} ± {agg['precision_std']:.3f}",
            "Recall":     f"{agg['recall_mean']:.3f} ± {agg['recall_std']:.3f}",
            "G-Mean":     f"{agg['g_mean_mean']:.3f} ± {agg['g_mean_std']:.3f}",
        })
        print(f"  {key:20s}  F1={agg['f1_macro_mean']:.3f}±{agg['f1_macro_std']:.3f}"
              f"  G-Mean={agg['g_mean_mean']:.3f}±{agg['g_mean_std']:.3f}")

    results_df = pd.DataFrame(rows)
    pd.DataFrame(selected).to_csv(os.path.join(RESULTS_DIR, "selected_hyperparameters.csv"), index=False)
    pd.DataFrame(f1_per_run).to_csv(os.path.join(RESULTS_DIR, "f1_per_run.csv"), index_label="run")
    from sklearn.metrics import precision_recall_fscore_support
    per = np.array([precision_recall_fscore_support(t, p, labels=sorted(CLASS_NAMES), zero_division=0)[:3]
                    for t, p in kb_runs])          # (run, metrica, classe)
    kb_rows = []
    for k, c in enumerate(sorted(CLASS_NAMES)):
        row = {"classe": CLASS_NAMES[c]}
        for m, nm in enumerate(["precision", "recall", "f1"]):
            row[nm + "_mean"] = per[:, m, k].mean(); row[nm + "_std"] = per[:, m, k].std(ddof=1)
        kb_rows.append(row)
    pd.DataFrame(kb_rows).round(3).to_csv(os.path.join(RESULTS_DIR, "kb_per_class.csv"), index=False)
    results_path = os.path.join(RESULTS_DIR, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[main] Risultati salvati in: {results_path}")

    # ── Test di Friedman ─────────────────────────────────────────────────────
    model_names = list(all_metrics.keys())
    f1_matrix = np.array([f1_per_run[m] for m in model_names])  # (n_models, n_runs)
    chi2, pval = friedman_test(f1_matrix)
    print(f"\n[main] Test di Friedman: χ²={chi2:.2f}, p={pval:.4e}")
    if pval < 0.05:
        print("       → Differenze statisticamente significative (p < 0.05)")
        pmat = nemenyi_pvalue_matrix(f1_matrix)
        plot_nemenyi_heatmap(pmat, model_names)
    else:
        print("       → Nessuna differenza significativa")

    # ── Grafici finali ────────────────────────────────────────────────────────
    plot_metrics_barplot(aggregated, model_names)
    plot_f1_boxplot(f1_per_run)

    print("\n[main] Pipeline completata. Risultati in:", RESULTS_DIR)


if __name__ == "__main__":
    run_pipeline()
