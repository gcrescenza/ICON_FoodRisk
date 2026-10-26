"""
src/bernoulli_nb.py
Bernoulli Naive Bayes implementato da zero (senza sklearn).
Lavora su feature binarie/discretizzate (feature OWL 0/1).
"""

import numpy as np
from config import BNB_ALPHA, N_CLASSES


class BernoulliNB:
    """
    Bernoulli Naive Bayes con Laplace smoothing.

    P(c)         = prior di classe
    P(f_i=1 | c) = probabilità condizionale per feature i in classe c
    log P(x | c) = sum_i [ x_i * log p_ic + (1-x_i) * log(1-p_ic) ]
    """

    def __init__(self, alpha: float = BNB_ALPHA, n_classes: int = N_CLASSES):
        self.alpha = alpha
        self.n_classes = n_classes
        self.log_prior = None       # (n_classes,)
        self.log_prob_pos = None    # (n_classes, n_features)
        self.log_prob_neg = None    # (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, n_feat = X.shape
        self.log_prior    = np.zeros(self.n_classes)
        self.log_prob_pos = np.zeros((self.n_classes, n_feat))
        self.log_prob_neg = np.zeros((self.n_classes, n_feat))

        for c in range(self.n_classes):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            self.log_prior[c] = np.log((n_c + self.alpha) /
                                       (n + self.alpha * self.n_classes))
            # P(f_i=1 | c) con Laplace smoothing
            p_pos = (X_c.sum(axis=0) + self.alpha) / (n_c + 2 * self.alpha)
            self.log_prob_pos[c] = np.log(p_pos)
            self.log_prob_neg[c] = np.log(1.0 - p_pos)
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Restituisce log-probabilità non normalizzate (n_samples × n_classes)."""
        # X: (n, d),  log_prob_pos: (K, d)
        log_pos = X @ self.log_prob_pos.T          # (n, K)
        log_neg = (1 - X) @ self.log_prob_neg.T   # (n, K)
        return self.log_prior + log_pos + log_neg  # (n, K)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_p = self.predict_log_proba(X)
        # softmax numericamente stabile
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_log_proba(X), axis=1)

    def get_params(self):
        return {"alpha": self.alpha, "n_classes": self.n_classes}
