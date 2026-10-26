"""
data/generate_dataset.py
Genera un dataset sintetico di ~900 pasti per 5 profili di rischio/idoneità.

NOTA METODOLOGICA: le deviazioni standard sono volutamente ampie e le medie
tra classi vicine parzialmente sovrapposte, in modo da NON rendere il problema
linearmente separabile sulle sole feature raw. Questo lascia margine reale di
miglioramento alle feature di Background Knowledge (OWL), che catturano
combinazioni congiunte di soglie più robuste al rumore della singola feature.
Un dataset "troppo pulito" (classi perfettamente separate da poche feature)
violerebbe lo spirito delle linee guida ICon, che richiedono di mostrare un
apporto genuino e misurabile della rappresentazione della conoscenza.
"""

import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATASET_PATH, RANDOM_SEED, N_CLASSES

rng = np.random.default_rng(RANDOM_SEED)


def generate_class(label: int, n: int) -> pd.DataFrame:
    """
    Ogni profilo ha distribuzioni caratteristiche sulle feature nutrizionali.
    Parametri scelti manualmente sulla base di linee guida nutrizionali generiche
    (es. soglie indicative su sodio/zuccheri/indice glicemico) per rendere le
    classi plausibili ma con sovrapposizione realistica e rumore intra-classe
    significativo (std elevate rispetto alla separazione tra le medie).
    """
    profiles = {
        # Bilanciato: tutti i valori in range medio, nessun eccesso marcato
        0: dict(
            calories                = rng.normal(0.48, 0.18, n).clip(0.05, 0.95),
            sodium_mg               = rng.normal(0.40, 0.18, n).clip(0.0, 0.95),
            sugar_g                 = rng.normal(0.35, 0.16, n).clip(0.0, 0.9),
            saturated_fat_g         = rng.normal(0.35, 0.16, n).clip(0.0, 0.9),
            fiber_g                 = rng.normal(0.50, 0.20, n).clip(0.0, 1.0),
            protein_g               = rng.normal(0.48, 0.17, n).clip(0.0, 0.95),
            glycemic_index          = rng.normal(0.45, 0.17, n).clip(0.0, 0.95),
            n_processed_ingredients = rng.poisson(2.0, n).clip(0, 8),
            potassium_mg            = rng.normal(0.50, 0.18, n).clip(0.0, 1.0),
            meal_type                = rng.choice([0, 1, 2, 3], n, p=[0.25, 0.30, 0.30, 0.15]),
        ),
        # Diabete-friendly: basso indice glicemico, zuccheri bassi, fibre alte
        1: dict(
            calories                = rng.normal(0.42, 0.18, n).clip(0.05, 0.9),
            sodium_mg               = rng.normal(0.38, 0.18, n).clip(0.0, 0.9),
            sugar_g                 = rng.normal(0.24, 0.15, n).clip(0.0, 0.75),
            saturated_fat_g         = rng.normal(0.32, 0.16, n).clip(0.0, 0.85),
            fiber_g                 = rng.normal(0.66, 0.18, n).clip(0.1, 1.0),
            protein_g               = rng.normal(0.50, 0.17, n).clip(0.0, 0.95),
            glycemic_index          = rng.normal(0.30, 0.16, n).clip(0.0, 0.85),
            n_processed_ingredients = rng.poisson(1.3, n).clip(0, 6),
            potassium_mg            = rng.normal(0.56, 0.18, n).clip(0.0, 1.0),
            meal_type                = rng.choice([0, 1, 2, 3], n, p=[0.30, 0.30, 0.30, 0.10]),
        ),
        # Iperteso-friendly: sodio basso, potassio adeguato
        2: dict(
            calories                = rng.normal(0.45, 0.18, n).clip(0.05, 0.9),
            sodium_mg               = rng.normal(0.24, 0.16, n).clip(0.0, 0.75),
            sugar_g                 = rng.normal(0.34, 0.17, n).clip(0.0, 0.9),
            saturated_fat_g         = rng.normal(0.32, 0.16, n).clip(0.0, 0.85),
            fiber_g                 = rng.normal(0.50, 0.20, n).clip(0.0, 1.0),
            protein_g               = rng.normal(0.46, 0.17, n).clip(0.0, 0.95),
            glycemic_index          = rng.normal(0.45, 0.17, n).clip(0.0, 0.95),
            n_processed_ingredients = rng.poisson(1.4, n).clip(0, 6),
            potassium_mg            = rng.normal(0.66, 0.18, n).clip(0.1, 1.0),
            meal_type                = rng.choice([0, 1, 2, 3], n, p=[0.25, 0.30, 0.35, 0.10]),
        ),
        # Sportivo: alte proteine, calorie elevate, basso grasso saturo
        3: dict(
            calories                = rng.normal(0.65, 0.18, n).clip(0.15, 1.0),
            sodium_mg               = rng.normal(0.42, 0.18, n).clip(0.0, 0.95),
            sugar_g                 = rng.normal(0.38, 0.17, n).clip(0.0, 0.95),
            saturated_fat_g         = rng.normal(0.28, 0.15, n).clip(0.0, 0.8),
            fiber_g                 = rng.normal(0.48, 0.19, n).clip(0.0, 1.0),
            protein_g               = rng.normal(0.74, 0.16, n).clip(0.15, 1.0),
            glycemic_index          = rng.normal(0.48, 0.17, n).clip(0.0, 0.95),
            n_processed_ingredients = rng.poisson(1.6, n).clip(0, 7),
            potassium_mg            = rng.normal(0.57, 0.18, n).clip(0.0, 1.0),
            meal_type                = rng.choice([0, 1, 2, 3], n, p=[0.30, 0.30, 0.25, 0.15]),
        ),
        # Ipercalorico/rischio: alti grassi saturi, alto sodio, alti zuccheri, processato
        4: dict(
            calories                = rng.normal(0.72, 0.18, n).clip(0.2, 1.0),
            sodium_mg               = rng.normal(0.64, 0.18, n).clip(0.1, 1.0),
            sugar_g                 = rng.normal(0.60, 0.18, n).clip(0.1, 1.0),
            saturated_fat_g         = rng.normal(0.64, 0.18, n).clip(0.1, 1.0),
            fiber_g                 = rng.normal(0.28, 0.16, n).clip(0.0, 0.8),
            protein_g               = rng.normal(0.42, 0.18, n).clip(0.0, 0.9),
            glycemic_index          = rng.normal(0.62, 0.18, n).clip(0.1, 1.0),
            n_processed_ingredients = rng.poisson(3.8, n).clip(0, 12),
            potassium_mg            = rng.normal(0.36, 0.17, n).clip(0.0, 0.9),
            meal_type                = rng.choice([0, 1, 2, 3], n, p=[0.15, 0.30, 0.30, 0.25]),
        ),
    }
    df = pd.DataFrame(profiles[label])
    df["label"] = label
    return df


def main():
    n_per_class = 180   # totale 900
    frames = [generate_class(c, n_per_class) for c in range(N_CLASSES)]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset generato: {DATASET_PATH}  ({len(df)} righe)")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
