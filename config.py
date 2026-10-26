# config.py — Configurazione globale FOODRISK

import os

# ─── Percorsi ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
ONTOLOGY_DIR = os.path.join(BASE_DIR, "ontology")
KB_DIR       = os.path.join(BASE_DIR, "kb")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

DATASET_PATH   = os.path.join(DATA_DIR, "meals.csv")
ONTOLOGY_PATH  = os.path.join(ONTOLOGY_DIR, "food_ontology.owl")
PROLOG_KB_PATH = os.path.join(KB_DIR, "food_kb.pl")

# ─── Classi target ───────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Bilanciato",
    1: "Diabete-friendly",
    2: "Iperteso-friendly",
    3: "Sportivo",
    4: "Ipercalorico/Rischio",
}
N_CLASSES = len(CLASS_NAMES)

# ─── Feature raw ─────────────────────────────────────────────────────────────
RAW_FEATURES = [
    "calories",
    "sodium_mg",
    "sugar_g",
    "saturated_fat_g",
    "fiber_g",
    "protein_g",
    "glycemic_index",
    "n_processed_ingredients",
    "potassium_mg",
    "meal_type",
]

# Costanti di dominio per la scalatura (non dipendono dal dataset di training):
# massimo conteggio di ingredienti processati e massimo codice di meal_type.
N_PROCESSED_MAX = 9
MEAL_TYPE_MAX   = 3

# ─── Valutazione ─────────────────────────────────────────────────────────────
N_RUNS        = 10
TEST_SIZE     = 0.30
RANDOM_SEED   = 42

# ─── Iperparametri modelli ───────────────────────────────────────────────────
# DT_PARAMS ecc. sono i valori di partenza. Il parametro principale di ogni
# modello e' scelto in ogni run per cross-validation (5 fold) sul solo training
# set, sulla griglia TUNING_GRIDS (src/supervised.tuned_fit).
DT_PARAMS  = {"max_depth": 5, "min_samples_split": 5, "random_state": RANDOM_SEED}
RF_PARAMS  = {"n_estimators": 100, "max_depth": 12, "random_state": RANDOM_SEED}
LR_PARAMS  = {"max_iter": 500, "random_state": RANDOM_SEED, "C": 1.0}
SVM_PARAMS = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True}

TUNING_GRIDS = {
    "DecisionTree":       {"max_depth": [3, 5, 8, 10, 15]},
    "RandomForest":       {"max_depth": [4, 8, 12, None]},
    "LogisticRegression": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "SVM_RBF":            {"C": [0.1, 1.0, 10.0, 100.0]},
}

# ─── BNB / EM ────────────────────────────────────────────────────────────────
BNB_ALPHA  = 1.0       # Laplace smoothing
EM_MAX_ITER = 20
EM_TOL      = 1e-4
UNLABELED_RATIO = 0.40  # Frazione dati usata come non etichettata nell'EM

# ─── Rete Bayesiana ──────────────────────────────────────────────────────────
BN_ALPHA = 1.0          # Laplace smoothing delle CPD

# ─── Plot ─────────────────────────────────────────────────────────────────────
FIGURE_DPI = 150
PALETTE    = ["#1A5E3C", "#2E86AB", "#A23B72", "#F18F01", "#44A0BA"]
