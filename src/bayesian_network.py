"""
src/bayesian_network.py
Rete Bayesiana discreta sulle feature BK (OWL) binarie, implementata con NumPy
(senza pgmpy), con inferenza esatta.

Struttura (DAG):
  profile -> bk_i                      per ognuno dei 12 concetti BK
  metabolic_risk        -> ultra_processed_heavy   (arco inter-feature)
  diabetes_compatible   -> low_glycemic_load       (arco inter-feature)

Fattorizzazione:  P(c, x) = P(c) * prod_i P(x_i | c, pa_extra(i))
dove pa_extra(i) e' l'eventuale genitore aggiuntivo del nodo i.

Apprendimento: stima di massima verosimiglianza con smoothing di Laplace.
Inferenza: con tutte le feature osservate, la distribuzione a posteriori su
`profile` si ottiene in forma chiusa normalizzando il prodotto dei fattori
(equivalente all'eliminazione di variabili sul solo nodo target).
"""

import numpy as np
from config import N_CLASSES, BN_ALPHA
from src.ontology_features import OWL_FEATURE_NAMES

# Archi inter-feature (genitore -> figlio), motivati dall'ontologia
EXTRA_EDGES = [
    ("bk_metabolic_risk",      "bk_ultra_processed_heavy"),
    ("bk_diabetes_compatible", "bk_low_glycemic_load"),
]


class FoodBayesianNetwork:
    """Rete Bayesiana (Naive Bayes aumentata) per il profilo nutrizionale."""

    TARGET = "profile"

    def __init__(self, alpha: float = BN_ALPHA, n_classes: int = N_CLASSES,
                 feature_names=None):
        self.alpha = alpha
        self.n_classes = n_classes
        self.feature_names = list(feature_names or OWL_FEATURE_NAMES)
        self._idx = {n: i for i, n in enumerate(self.feature_names)}
        # figlio -> indice del genitore aggiuntivo
        self._extra_parent = {self._idx[c]: self._idx[p] for p, c in EXTRA_EDGES
                              if p in self._idx and c in self._idx}
        self.log_prior = None
        self.cpd = {}          # i -> array (n_classes, n_parent_states, 2) di log P(x_i | c, pa)
        self._fitted = False

    # ── struttura ────────────────────────────────────────────────────────────
    def edges(self):
        e = [(self.TARGET, f) for f in self.feature_names]
        e += [(p, c) for p, c in EXTRA_EDGES
              if p in self._idx and c in self._idx]
        return e

    # ── apprendimento ────────────────────────────────────────────────────────
    def fit(self, X_bin: np.ndarray, y: np.ndarray):
        X = np.asarray(X_bin).astype(int)
        n, d = X.shape
        K, a = self.n_classes, self.alpha
        counts = np.bincount(y, minlength=K)
        self.log_prior = np.log((counts + a) / (n + a * K))

        for i in range(d):
            if i in self._extra_parent:
                pa = X[:, self._extra_parent[i]]
                n_states = 2
            else:
                pa = np.zeros(n, dtype=int)
                n_states = 1
            table = np.zeros((K, n_states, 2))
            for c in range(K):
                for s in range(n_states):
                    m = (y == c) & (pa == s)
                    n1 = X[m, i].sum()
                    nt = m.sum()
                    p1 = (n1 + a) / (nt + 2 * a)
                    table[c, s, 1] = np.log(p1)
                    table[c, s, 0] = np.log(1.0 - p1)
            self.cpd[i] = table
        self._fitted = True
        return self

    # ── inferenza ────────────────────────────────────────────────────────────
    def predict_log_joint(self, X_bin: np.ndarray) -> np.ndarray:
        X = np.asarray(X_bin).astype(int)
        n, d = X.shape
        out = np.tile(self.log_prior, (n, 1))
        for i in range(d):
            s = X[:, self._extra_parent[i]] if i in self._extra_parent else np.zeros(n, dtype=int)
            out += self.cpd[i][:, s, X[:, i]].T
        return out

    def predict_proba(self, X_bin: np.ndarray) -> np.ndarray:
        """Distribuzione a posteriori P(profile | feature BK osservate)."""
        lj = self.predict_log_joint(X_bin)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X_bin: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_log_joint(X_bin), axis=1)

    def posterior(self, evidence: dict) -> np.ndarray:
        """
        Posteriori per una singola istanza data come {nome_feature_bk: 0/1}.
        Le feature non indicate sono poste a 0.
        """
        x = np.zeros((1, len(self.feature_names)))
        for k, v in evidence.items():
            x[0, self._idx[k]] = v
        return self.predict_proba(x)[0]

    @property
    def available(self):
        return True
