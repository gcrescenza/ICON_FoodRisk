"""
robustness_check.py — Valutazione fuori campione su dataset sintetici nuovi.

I modelli (iperparametri scelti per CV sul training originale) sono addestrati
sull'intero data/meals.csv e valutati su dataset generati con seed diversi in
due scenari:
  A) stesso generatore (5 seed): sensibilita' al campionamento;
  B) generatore PERTURBATO (5 seed): le deviazioni standard di tutte le
     distribuzioni normali sono moltiplicate per NOISE_FACTOR, cioe' piu'
     sovrapposizione tra classi: una distribuzione diversa da quella su cui
     sono state definite soglie BK/Prolog e calibrati i modelli.
Resta un test su dati sintetici: non misura la generalizzazione a pasti reali.

Uso:  python robustness_check.py   ->  results/independent_check.csv
"""

import os
import numpy as np
import pandas as pd

import data.generate_dataset as gen
from config import RESULTS_DIR, RAW_FEATURES, N_CLASSES, RANDOM_SEED
from src.preprocessing import load_data, scale_features
from src.supervised import tuned_fit, get_models
from src.ontology_features import extract_owl_features, augment_with_owl
from src.bernoulli_nb import BernoulliNB
from src.bayesian_network import FoodBayesianNetwork
from src.prolog_kb import PrologKB
from src.evaluation import compute_metrics

SCENARIOS = {"A: stesso generatore": (1.0, [101, 102, 103, 104, 105]),
             "B: rumore x1.3":        (1.3, [201, 202, 203, 204, 205])}


class _ScaledNoiseRng:
    """Wrapper di np.random.Generator che moltiplica la std delle normali."""
    def __init__(self, seed, factor):
        self._r, self._f = np.random.default_rng(seed), factor
    def normal(self, loc, scale, size=None):
        return self._r.normal(loc, scale * self._f, size)
    def __getattr__(self, name):
        return getattr(self._r, name)


def make_dataset(seed, factor):
    gen.rng = _ScaledNoiseRng(seed, factor)          # il generatore usa un rng di modulo
    df = pd.concat([gen.generate_class(c, 180) for c in range(N_CLASSES)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    train = load_data()
    X_all, y_all = train[RAW_FEATURES].values, train["label"].values
    Xtr, _ = scale_features(X_all, X_all[:5])        # scaler appreso sul training originale

    models = {}
    for fs, fn in (("raw", lambda X: X), ("owl", augment_with_owl)):
        for name in get_models():
            short = (name.replace("LogisticRegression", "LR").replace("_RBF", "")
                         .replace("DecisionTree", "DT").replace("RandomForest", "RF"))
            m, _ = tuned_fit(name, fn(Xtr), y_all, seed=RANDOM_SEED)
            models[f"{short}_{fs}"] = (m, fn)

    Xb_tr = (extract_owl_features(Xtr) >= 0.5).astype(float)
    bnb = BernoulliNB().fit(Xb_tr, y_all)
    bn = FoodBayesianNetwork().fit(Xb_tr, y_all)
    kb = PrologKB()

    table = {}
    for label, (factor, seeds) in SCENARIOS.items():
        res = {}
        for sd in seeds:
            t = make_dataset(sd, factor)
            _, Xte = scale_features(X_all, t[RAW_FEATURES].values)   # scaler del training originale
            yte = t["label"].values
            for k, (m, fn) in models.items():
                res.setdefault(k, []).append(compute_metrics(yte, m.predict(fn(Xte)))["f1_macro"])
            Xb = (extract_owl_features(Xte) >= 0.5).astype(float)
            res.setdefault("BNB_owl", []).append(compute_metrics(yte, bnb.predict(Xb))["f1_macro"])
            res.setdefault("BayesNet", []).append(compute_metrics(yte, bn.predict(Xb))["f1_macro"])
            res.setdefault("KB_Prolog", []).append(compute_metrics(yte, kb.infer_profile(Xte))["f1_macro"])
        for k, v in res.items():
            table.setdefault(k, {})[label] = f"{np.mean(v):.3f} ± {np.std(v):.3f}"

    out = pd.DataFrame([{"Modello": k, **v} for k, v in table.items()])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "independent_check.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\n[robustness_check] Salvato: {path}")


if __name__ == "__main__":
    main()
