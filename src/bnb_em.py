"""
src/bnb_em.py
Apprendimento semi-supervisionato: BNB con algoritmo EM.

Schema:
  1. E-step: calcola le responsabilità P(c | x_u) sui dati non etichettati
             usando il modello corrente
  2. M-step: aggiorna i parametri del BNB usando sia i dati etichettati
             (con peso 1) sia i dati non etichettati (con peso = responsabilità)
  Ripete fino a convergenza (variazione log-likelihood < tol).
"""

import numpy as np
from src.bernoulli_nb import BernoulliNB
from config import EM_MAX_ITER, EM_TOL, UNLABELED_RATIO, N_CLASSES, BNB_ALPHA


class BernoulliNB_EM:

    def __init__(self, alpha: float = BNB_ALPHA, n_classes: int = N_CLASSES,
                 max_iter: int = EM_MAX_ITER, tol: float = EM_TOL):
        self.alpha = alpha
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.model = BernoulliNB(alpha=alpha, n_classes=n_classes)
        self.n_iter_ = 0
        self.ll_history_ = []

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray,
            X_unlabeled: np.ndarray):
        """
        X_labeled:   dati con etichetta (n_l × d)
        y_labeled:   etichette (n_l,)
        X_unlabeled: dati privi di etichetta (n_u × d)
        """
        # Inizializzazione: addestra solo sui dati etichettati
        self.model.fit(X_labeled, y_labeled)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            # ── E-step ──────────────────────────────────────────────────────
            # Responsabilità sui dati non etichettati: (n_u, K)
            R = self.model.predict_proba(X_unlabeled)

            # ── M-step ──────────────────────────────────────────────────────
            n_l, n_feat = X_labeled.shape
            n_u = X_unlabeled.shape[0]

            log_prior    = np.zeros(self.n_classes)
            log_prob_pos = np.zeros((self.n_classes, n_feat))
            log_prob_neg = np.zeros((self.n_classes, n_feat))

            for c in range(self.n_classes):
                # Conteggi etichettati
                mask_l = (y_labeled == c)
                count_l = mask_l.sum()
                sum_l   = X_labeled[mask_l].sum(axis=0) if count_l > 0 else np.zeros(n_feat)

                # Conteggi pesati non etichettati
                r_c     = R[:, c]                         # (n_u,)
                count_u = r_c.sum()
                sum_u   = r_c @ X_unlabeled               # (d,)

                total_count = count_l + count_u
                total_sum   = sum_l + sum_u

                # Prior aggiornato
                log_prior[c] = np.log(
                    (total_count + self.alpha) /
                    (n_l + n_u + self.alpha * self.n_classes)
                )

                # Probabilità condizionali con Laplace smoothing
                p_pos = (total_sum + self.alpha) / (total_count + 2 * self.alpha)
                p_pos = np.clip(p_pos, 1e-10, 1 - 1e-10)
                log_prob_pos[c] = np.log(p_pos)
                log_prob_neg[c] = np.log(1.0 - p_pos)

            self.model.log_prior    = log_prior
            self.model.log_prob_pos = log_prob_pos
            self.model.log_prob_neg = log_prob_neg

            # ── Convergenza: log-likelihood sui dati etichettati ─────────────
            lp = self.model.predict_log_proba(X_labeled)
            ll = lp[np.arange(n_l), y_labeled].mean()
            self.ll_history_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                self.n_iter_ = it + 1
                break
            prev_ll = ll

        else:
            self.n_iter_ = self.max_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def split_labeled_unlabeled(X_train: np.ndarray, y_train: np.ndarray,
                             unlabeled_ratio: float = UNLABELED_RATIO,
                             seed: int = 42):
    """
    Simula lo scenario semi-supervisionato: nasconde le etichette di una
    frazione `unlabeled_ratio` dei dati di training.
    """
    rng = np.random.default_rng(seed)
    n = len(y_train)
    idx_unlabeled = rng.choice(n, int(n * unlabeled_ratio), replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx_unlabeled] = True

    X_labeled   = X_train[~mask]
    y_labeled   = y_train[~mask]
    X_unlabeled = X_train[mask]

    return X_labeled, y_labeled, X_unlabeled
