"""
src/prolog_kb.py
Interfaccia Python ↔ SWI-Prolog per la KB food_kb.pl.
Usa pyswip se disponibile; altrimenti implementa la logica in Python
come fallback per garantire l'esecuzione dell'intero pipeline.
"""

import numpy as np
import os
import shutil
import subprocess
import tempfile
from config import PROLOG_KB_PATH, RAW_FEATURES

try:
    from pyswip import Prolog
    PROLOG_AVAILABLE = True
except (ImportError, OSError):
    PROLOG_AVAILABLE = False


def find_swipl():
    """Percorso dell'eseguibile SWI-Prolog (variabile d'ambiente SWIPL o PATH), o None."""
    return os.environ.get("SWIPL") or shutil.which("swipl")


SWIPL_PATH = find_swipl()

# nome del predicato meal_* per ogni feature raw
_FACT_OF = {"calories": "meal_calories", "sodium_mg": "meal_sodium", "sugar_g": "meal_sugar",
            "saturated_fat_g": "meal_saturated_fat", "fiber_g": "meal_fiber",
            "protein_g": "meal_protein", "glycemic_index": "meal_glycemic_index",
            "potassium_mg": "meal_potassium"}


def swipl_batch(kb_path, X_raw, goal_template):
    """
    Esegue la KB con SWI-Prolog (sottoprocesso, senza pyswip) su tutti i pasti di X_raw.
    I fatti meal_*/2 sono scritti con la precisione piena dei float; per ogni pasto m<i>
    si valuta `goal_template` (es. "profile({M}, R)") e si stampa il primo valore di R.
    Restituisce la lista dei risultati come stringhe ("none" se il goal fallisce).
    """
    facts = []
    for i, row_vec in enumerate(X_raw):
        row = dict(zip(RAW_FEATURES, row_vec))
        for feat, pred in _FACT_OF.items():
            facts.append(f"{pred}(m{i}, {float(row[feat])!r}).")
    goal = goal_template.format(M="M")
    kb_path = os.path.abspath(kb_path).replace("\\", "/")  # Windows: '\\' e' un escape in Prolog
    prog = (f":- style_check(-discontiguous).\n:- consult('{kb_path}').\n"
            + "\n".join(facts)
            + f"\nrun :- forall(between(0, {len(X_raw) - 1}, I), "
              f"( atom_concat(m, I, M), ( once(({goal})) -> true ; R = none ), "
              f"format('~w~n', [R]) )).\n")
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as f:
        f.write(prog); tmp = f.name
    try:
        out = subprocess.run([SWIPL_PATH, "-q", "-f", "none", "-s", tmp, "-g", "run", "-t", "halt"],
                             capture_output=True, text=True, timeout=600, check=True,
                             env={**os.environ, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    finally:
        os.unlink(tmp)
    res = out.stdout.split()
    if len(res) != len(X_raw):
        raise RuntimeError(f"SWI-Prolog ha restituito {len(res)} risultati per {len(X_raw)} pasti")
    return res


# ─── Implementazione Python del ragionamento Prolog (fallback) ───────────────
# Le regole sono DATI (pesi, livelli, priorita'), cosi' che kb/verify_prolog.py
# possa confrontarle automaticamente con food_kb.pl. Le soglie dei livelli
# NON sono definite qui: sono lette dall'ontologia (src/ontology_features.LEVELS).

from src.ontology_features import LEVELS

# attributo Prolog -> feature raw
FEATURE_OF = {
    "sugar": "sugar_g", "sodium": "sodium_mg", "fiber": "fiber_g",
    "protein": "protein_g", "fat": "saturated_fat_g", "potassium": "potassium_mg",
    "calorie": "calories", "glycemic": "glycemic_index",
}

# profilo -> (costante, [(attributo, livello, peso)])   (peso negativo = penalita')
SCORE_RULES = {
    "diabete":      (0, [("sugar", "low", 2), ("fiber", "high", 2), ("glycemic", "low", 2),
                         ("glycemic", "medium", 1), ("sodium", "high", -1), ("calorie", "high", -1)]),
    "iperteso":     (0, [("sodium", "low", 2), ("potassium", "high", 2), ("potassium", "medium", 1),
                         ("protein", "high", -1)]),
    "sportivo":     (0, [("protein", "high", 3), ("fat", "low", 1), ("calorie", "high", 1)]),
    "ipercalorico": (0, [("fat", "high", 2), ("sodium", "high", 1), ("sugar", "high", 1),
                         ("calorie", "high", 1)]),
    "bilanciato":   (1, [("sugar", "medium", 1), ("sodium", "medium", 1), ("fat", "low", 1),
                         ("glycemic", "medium", 1), ("calorie", "high", -1)]),
}
PRIORITY = ["diabete", "iperteso", "sportivo", "ipercalorico", "bilanciato"]
PROFILE_ID = {"bilanciato": 0, "diabete": 1, "iperteso": 2, "sportivo": 3, "ipercalorico": 4}

# condizione -> [(attributo, livello, negato)]   (congiunzione, come compatible/2)
COMPATIBLE_RULES = {
    "diabete":        [("sugar", "low", False), ("glycemic", "low", False)],
    "iperteso":       [("sodium", "low", False), ("potassium", "low", True)],
    "sportivo":       [("protein", "high", False), ("fat", "low", False)],
    "metabolic_risk": [("fat", "low", False), ("sodium", "low", False), ("sugar", "low", False)],
}


def level_of(attr: str, value: float) -> str:
    """Livello low/medium/high di un attributo, con soglie lette dall'ontologia."""
    low, high = LEVELS[FEATURE_OF[attr]]
    return "low" if value < low else ("high" if value >= high else "medium")


def _infer_profile_python(row: dict) -> int:
    """
    Replica la logica a voto di food_kb.pl / profile/2: per ogni profilo somma i
    pesi dei livelli soddisfatti (SCORE_RULES); vince il punteggio massimo, con
    priorita' clinica in caso di pareggio (PRIORITY = priority_order/1).
    """
    lv = {a: level_of(a, row[f]) for a, f in FEATURE_OF.items()}
    scores = {p: base + sum(w for a, l, w in rules if lv[a] == l)
              for p, (base, rules) in SCORE_RULES.items()}
    best = max(scores.values())
    for p in PRIORITY:
        if scores[p] == best:
            return PROFILE_ID[p]
    return 0


def _is_compatible_python(meal: dict, condition: str) -> bool:
    """Compatibilita' pasto-condizione (COMPATIBLE_RULES = compatible/2 in food_kb.pl)."""
    rules = COMPATIBLE_RULES.get(condition)
    if rules is None:
        return False
    return all((level_of(a, meal[FEATURE_OF[a]]) == l) != neg for a, l, neg in rules)


# ─── Classe principale ────────────────────────────────────────────────────────

class PrologKB:

    def __init__(self, kb_path: str = PROLOG_KB_PATH):
        self.kb_path = kb_path
        self.prolog = None
        self._loaded = False
        self._swipl = False

        if not PROLOG_AVAILABLE and SWIPL_PATH and os.path.exists(kb_path):
            self._swipl = self._loaded = True
            print(f"[PrologKB] SWI-Prolog eseguito come sottoprocesso ({SWIPL_PATH}).")
        elif PROLOG_AVAILABLE and os.path.exists(kb_path):
            try:
                self.prolog = Prolog()
                self.prolog.consult(kb_path)
                self._loaded = True
                print("[PrologKB] SWI-Prolog caricato con successo.")
            except Exception as e:
                print(f"[PrologKB] Errore caricamento Prolog: {e} — uso fallback Python.")
        else:
            print("[PrologKB] pyswip / SWI-Prolog non disponibile — uso fallback Python.")

    def infer_profile(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Dato X_raw (n × n_raw_features), restituisce array di profili inferiti
        dalla KB per ogni pasto.
        """
        if self._swipl:
            try:
                res = swipl_batch(self.kb_path, X_raw, "profile({M}, R)")
                return np.array([PROFILE_ID.get(r, 0) for r in res])
            except Exception as e:
                print(f"[PrologKB] Errore SWI-Prolog: {e} — uso fallback Python.")
                self._swipl = self._loaded = False
        results = []
        for row_vec in X_raw:
            row = dict(zip(RAW_FEATURES, row_vec))

            if self._loaded and not self._swipl:
                meal_id = f"m{id(row_vec)}"
                try:
                    # Asserisce i fatti per questo pasto
                    self.prolog.assertz(f"meal_calories({meal_id}, {row['calories']:.4f})")
                    self.prolog.assertz(f"meal_sodium({meal_id}, {row['sodium_mg']:.4f})")
                    self.prolog.assertz(f"meal_sugar({meal_id}, {row['sugar_g']:.4f})")
                    self.prolog.assertz(f"meal_saturated_fat({meal_id}, {row['saturated_fat_g']:.4f})")
                    self.prolog.assertz(f"meal_fiber({meal_id}, {row['fiber_g']:.4f})")
                    self.prolog.assertz(f"meal_protein({meal_id}, {row['protein_g']:.4f})")
                    self.prolog.assertz(f"meal_glycemic_index({meal_id}, {row['glycemic_index']:.4f})")
                    self.prolog.assertz(f"meal_potassium({meal_id}, {row['potassium_mg']:.4f})")

                    profile_map = {"bilanciato": 0, "diabete": 1, "iperteso": 2,
                                   "sportivo": 3, "ipercalorico": 4}
                    pred = 0  # default
                    for sol in self.prolog.query(f"profile({meal_id}, P)"):
                        pred = profile_map.get(str(sol["P"]), 0)
                        break
                    results.append(pred)

                    # Ritrae i fatti per evitare accumulo
                    self.prolog.retract(f"meal_calories({meal_id}, {row['calories']:.4f})")
                    self.prolog.retract(f"meal_sodium({meal_id}, {row['sodium_mg']:.4f})")
                    self.prolog.retract(f"meal_sugar({meal_id}, {row['sugar_g']:.4f})")
                    self.prolog.retract(f"meal_saturated_fat({meal_id}, {row['saturated_fat_g']:.4f})")
                    self.prolog.retract(f"meal_fiber({meal_id}, {row['fiber_g']:.4f})")
                    self.prolog.retract(f"meal_protein({meal_id}, {row['protein_g']:.4f})")
                    self.prolog.retract(f"meal_glycemic_index({meal_id}, {row['glycemic_index']:.4f})")
                    self.prolog.retract(f"meal_potassium({meal_id}, {row['potassium_mg']:.4f})")
                except Exception:
                    results.append(_infer_profile_python(row))
            else:
                results.append(_infer_profile_python(row))

        return np.array(results)

    def check_compatibility(self, meal: dict, condition: str) -> bool:
        """Verifica compatibilità pasto-condizione tramite la KB."""
        if self._loaded:
            mid = "meal1"
            try:
                self.prolog.assertz(f"meal_sugar({mid}, {meal['sugar_g']:.4f})")
                self.prolog.assertz(f"meal_glycemic_index({mid}, {meal['glycemic_index']:.4f})")
                self.prolog.assertz(f"meal_sodium({mid}, {meal['sodium_mg']:.4f})")
                self.prolog.assertz(f"meal_potassium({mid}, {meal['potassium_mg']:.4f})")
                self.prolog.assertz(f"meal_protein({mid}, {meal['protein_g']:.4f})")
                self.prolog.assertz(f"meal_saturated_fat({mid}, {meal['saturated_fat_g']:.4f})")

                result = bool(list(self.prolog.query(f"compatible({mid}, {condition})")))

                # pulizia
                for fact in [f"meal_sugar({mid}, _)", f"meal_glycemic_index({mid}, _)",
                             f"meal_sodium({mid}, _)", f"meal_potassium({mid}, _)",
                             f"meal_protein({mid}, _)", f"meal_saturated_fat({mid}, _)"]:
                    try:
                        self.prolog.retract(fact)
                    except Exception:
                        pass
                return result
            except Exception:
                pass
        return _is_compatible_python(meal, condition)

    @property
    def available(self):
        return self._loaded
