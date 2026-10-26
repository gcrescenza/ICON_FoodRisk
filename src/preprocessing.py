"""
src/preprocessing.py
Caricamento, scalatura e split del dataset.

Scalatura (senza data leakage):
  - le feature continue (gia' nel range [0,1] per costruzione) sono scalate con
    un MinMaxScaler addestrato SOLO sul training set di ciascuno split;
  - `n_processed_ingredients` e `meal_type` sono scalate con costanti di dominio
    (N_PROCESSED_MAX, MEAL_TYPE_MAX di config.py), non apprese dai dati: cosi'
    le soglie dei concetti BK sul conteggio hanno un significato stabile.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from config import (DATASET_PATH, RAW_FEATURES, TEST_SIZE, RANDOM_SEED,
                    N_PROCESSED_MAX, MEAL_TYPE_MAX)

_IDX_PROC = RAW_FEATURES.index("n_processed_ingredients")
_IDX_MEAL = RAW_FEATURES.index("meal_type")
_IDX_CONT = [i for i in range(len(RAW_FEATURES)) if i not in (_IDX_PROC, _IDX_MEAL)]


def load_data() -> pd.DataFrame:
    """Carica il dataset CSV."""
    df = pd.read_csv(DATASET_PATH)
    print(f"[preprocessing] Caricati {len(df)} campioni, {df['label'].nunique()} classi")
    return df


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """
    Scala (X_train, X_test) senza leakage: lo scaler e' appreso su X_train.
    Restituisce (X_train_scaled, X_test_scaled).
    """
    Xtr = np.asarray(X_train, dtype=float).copy()
    Xte = np.asarray(X_test, dtype=float).copy()
    scaler = MinMaxScaler().fit(Xtr[:, _IDX_CONT])
    Xtr[:, _IDX_CONT] = scaler.transform(Xtr[:, _IDX_CONT])
    Xte[:, _IDX_CONT] = scaler.transform(Xte[:, _IDX_CONT])
    for X in (Xtr, Xte):
        X[:, _IDX_PROC] /= N_PROCESSED_MAX
        X[:, _IDX_MEAL] /= MEAL_TYPE_MAX
    return Xtr, Xte


def get_splits(df: pd.DataFrame, n_splits: int = 1, random_state: int = RANDOM_SEED):
    """
    Restituisce (lista di (X_train, X_test, y_train, y_test), None).
    Usa StratifiedShuffleSplit; la scalatura e' fatta dentro ogni split
    sul solo training set. Il secondo elemento e' mantenuto per
    compatibilita' con la firma precedente.
    """
    X = df[RAW_FEATURES].values
    y = df["label"].values

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=TEST_SIZE,
                                 random_state=random_state)
    splits = []
    for train_idx, test_idx in sss.split(X, y):
        Xtr, Xte = scale_features(X[train_idx], X[test_idx])
        splits.append((Xtr, Xte, y[train_idx], y[test_idx]))
    return splits, None
