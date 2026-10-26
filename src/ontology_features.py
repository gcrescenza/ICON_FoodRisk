"""
src/ontology_features.py
Feature di Background Knowledge derivate dall'ontologia OWL.

I 12 concetti BK (congiunzioni di 2-4 vincoli su feature diverse, es.
MetabolicRisk = grassi saturi alti AND sodio alto AND zuccheri alti) e i livelli
nutrizionali low/high NON sono scritti qui: vengono LETTI dal file
`ontology/food_ontology.owl` (generato da ontology/spec.py se assente) tramite
`ontology.owl_io.load_bk`. Questo modulo si limita a valutare le definizioni
lette sulle istanze.

Una soglia su una sola feature non aggiunge informazione che un albero o un SVM
non possano apprendere da soli; una congiunzione di piu' vincoli, derivata dalla
semantica clinica delle proprieta' `requiresLowLevelOf` / `requiresHighLevelOf`,
e' un'astrazione piu' ricca (ma resta una funzione deterministica delle feature
raw: vedi docs/relazione.md, sezione 9).

Scalatura: X_raw arriva da src/preprocessing.py (feature continue in [0,1];
n_processed_ingredients = conteggio / N_PROCESSED_MAX). Le soglie sul conteggio
(datatype xsd:integer) sono convertite a meta' intero ((t-0.5)/N_PROCESSED_MAX).
"""

import numpy as np
import pandas as pd
from config import RAW_FEATURES, N_PROCESSED_MAX, ONTOLOGY_PATH
from ontology.owl_io import ensure_ontology, load_bk

ensure_ontology(ONTOLOGY_PATH)
_LEVELS_FILE, _CONCEPTS_FILE, _DTYPES = load_bk(ONTOLOGY_PATH)

# Livelli nelle unita' delle feature scalate: feature -> (lowBelow, highFrom)
LEVELS = {f: lv for f, lv in _LEVELS_FILE.items() if _DTYPES[f] != "integer"}

# Definizioni dei concetti come lette dal file: nome -> [(feature, op, valore, dtype)]
BK_DEFINITIONS = _CONCEPTS_FILE


def _scaled_threshold(feature, op, value, dtype):
    """Converte (op, valore) nello spazio scalato; i conteggi interi usano il mezzo intero."""
    if dtype != "integer":
        return op, value
    if op in ("<", ">="):
        return op, (value - 0.5) / N_PROCESSED_MAX
    return (">=", (value + 0.5) / N_PROCESSED_MAX) if op == ">" else ("<", (value + 0.5) / N_PROCESSED_MAX)


def _make_concept(constraints):
    prepared = [(f,) + _scaled_threshold(f, op, v, dt) for f, op, v, dt in constraints]

    def fn(df):
        mask = np.ones(len(df), dtype=bool)
        for feat, op, thr in prepared:
            col = df[feat].values
            mask &= {"<": col < thr, ">=": col >= thr, "<=": col <= thr, ">": col > thr}[op]
        return pd.Series(mask.astype(float))
    return fn


BK_CONCEPTS = {name: _make_concept(cons) for name, cons in BK_DEFINITIONS.items()}
OWL_FEATURE_NAMES = list(BK_CONCEPTS.keys())


def extract_owl_features(X_raw: np.ndarray) -> np.ndarray:
    """X_raw (n x n_raw_features) -> X_owl (n x 12), valori 0/1."""
    df = pd.DataFrame(X_raw, columns=RAW_FEATURES)
    return np.stack([fn(df).values for fn in BK_CONCEPTS.values()], axis=1)


def augment_with_owl(X_raw: np.ndarray) -> np.ndarray:
    """Concatena feature raw + feature OWL."""
    return np.concatenate([X_raw, extract_owl_features(X_raw)], axis=1)


def owl_coverage_per_class(X_raw: np.ndarray, y: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Matrice (n_classes x 12): attivazione media di ogni concetto per classe."""
    X_owl = extract_owl_features(X_raw)
    cov = np.zeros((n_classes, len(BK_CONCEPTS)))
    for c in range(n_classes):
        m = (y == c)
        if m.sum() > 0:
            cov[c] = X_owl[m].mean(axis=0)
    return cov
