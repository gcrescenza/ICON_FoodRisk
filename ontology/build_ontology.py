"""
ontology/build_ontology.py
Genera `ontology/food_ontology.owl` dalla specifica `ontology/spec.py`
(nessuna dipendenza esterna) e ne stampa le statistiche leggendo il file.

Uso:  python ontology/build_ontology.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ONTOLOGY_PATH
from ontology.owl_io import write_owl, summary


def build_ontology():
    write_owl(ONTOLOGY_PATH)
    s = summary(ONTOLOGY_PATH)
    print(f"Ontologia salvata: {ONTOLOGY_PATH}")
    print(f"  Classi:           {s['classi']} (di cui {s['concetti_bk']} concetti BK definiti)")
    print(f"  Object property:  {s['object_property']}")
    print(f"  Data property:    {s['data_property']}")
    print(f"  Individui:        {s['individui']}")
    return s


if __name__ == "__main__":
    build_ontology()
