# FOODRISK — Meal Nutritional Risk Knowledge-Based Classification System

## Classificazione del profilo di rischio nutrizionale di un pasto con Ontologia e Apprendimento Probabilistico

Progetto di **Ingegneria della Conoscenza** — A.A. 2025/2026  
Università degli Studi di Bari Aldo Moro  
Autore: Giorgio Crescenza — matricola 698991 — g.crescenza2@studenti.uniba.it

---

## Descrizione

**FOODRISK** è un sistema intelligente per la **classificazione del profilo di rischio/idoneità nutrizionale** di un pasto rispetto a condizioni di salute target, integrando:

- Apprendimento supervisionato (Decision Tree, Random Forest, Logistic Regression, SVM)
- Apprendimento semi-supervisionato (Bernoulli NB + algoritmo EM)
- Background Knowledge Ontologica (OWL — `ontology/food_ontology.owl`: tassonomia, livelli nutrizionali e 12 concetti definiti da restrizioni; **letta dalla pipeline**)
- Knowledge Base rule-based (Prolog — `food_kb.pl`) per inferenza su compatibilità pasto-condizione
- Rete Bayesiana (NumPy, inferenza esatta) con posteriori sul profilo

Il sistema parte da un dataset di pasti (composizione nutrizionale + feature derivate) e ne predice la **categoria di idoneità clinico-nutrizionale** rispetto a 5 profili target, usando la conoscenza ontologica come Background Knowledge per arricchire le feature numeriche grezze.

---

## Dataset

Il dataset (`data/meals.csv`) contiene **900 pasti** (180 per classe) sinteticamente generati (composizione realistica ispirata a linee guida nutrizionali), con le seguenti feature:

| Feature                   | Tipo    | Descrizione                                            |
|----------------------------|---------|---------------------------------------------------------|
| `calories`                 | float   | Calorie totali (kcal, normalizzate)                     |
| `sodium_mg`                 | float   | Sodio (mg, normalizzato)                                 |
| `sugar_g`                   | float   | Zuccheri (g, normalizzato)                                |
| `saturated_fat_g`           | float   | Grassi saturi (g, normalizzato)                           |
| `fiber_g`                   | float   | Fibre (g, normalizzato)                                  |
| `protein_g`                 | float   | Proteine (g, normalizzato)                                |
| `glycemic_index`            | float   | Indice glicemico medio del pasto (0-1 norm.)              |
| `n_processed_ingredients`   | int     | N° ingredienti ultra-processati                          |
| `potassium_mg`              | float   | Potassio (mg, normalizzato)                              |
| `meal_type`                 | int     | Colazione=0, Pranzo=1, Cena=2, Spuntino=3                |
| `label`                    | int     | Profilo di rischio (0=Bilanciato, 1=Diabete, 2=Iperteso, 3=Sportivo, 4=Ipercalorico) |

**Classi target (idoneità del pasto rispetto al profilo):**

| Classe | Profilo              | Caratteristiche tipiche                                     |
|--------|------------------------|-----------------------------------------------------------------|
| 0      | Bilanciato             | Macro equilibrati, sodio/zuccheri moderati                    |
| 1      | Diabete-friendly        | Basso indice glicemico, zuccheri bassi, fibre alte             |
| 2      | Iperteso-friendly       | Sodio basso, potassio adeguato                                 |
| 3      | Sportivo                | Alte proteine, calorie elevate, basso grasso saturo            |
| 4      | Ipercalorico/rischio    | Alti grassi saturi, alto sodio, alti zuccheri, processato      |

---

## Struttura del Progetto

```
FOODRISK/
├── main.py                        # Pipeline principale (12 modelli, test statistici)
├── robustness_check.py            # Test su dataset nuovi e perturbati
├── config.py                      # Costanti, griglie di tuning, percorsi
├── requirements.txt
├── data/
│   ├── meals.csv                  # Dataset principale
│   └── generate_dataset.py        # Generatore dataset sintetico
├── ontology/
│   ├── spec.py                    # Specifica dichiarativa (fonte unica: classi, livelli, concetti BK)
│   ├── owl_io.py                  # Scrittura/lettura OWL (RDF/XML, solo libreria standard)
│   ├── build_ontology.py          # Genera food_ontology.owl da spec.py
│   └── food_ontology.owl          # Ontologia OWL (letta da src/ontology_features.py)
├── kb/
│   ├── food_kb.pl                 # Knowledge Base Prolog
│   └── verify_prolog.py           # Coerenza KB <-> ontologia <-> Python; esecuzione Prolog su 900 pasti
├── src/
│   ├── preprocessing.py           # Scalatura senza leakage, split
│   ├── ontology_features.py       # Legge dal file OWL livelli e concetti BK; calcola le feature
│   ├── supervised.py              # DT, RF, LR, SVM + tuning per CV
│   ├── bernoulli_nb.py            # BNB da zero
│   ├── bnb_em.py                  # BNB-EM semi-supervisionato
│   ├── bayesian_network.py        # Rete bayesiana su feature BK (NumPy, inferenza esatta)
│   ├── prolog_kb.py               # Interfaccia Python -> SWI-Prolog (sottoprocesso/pyswip) + riferimento Python
│   ├── evaluation.py              # Metriche, test Friedman, Nemenyi
│   └── visualization.py           # Plot e heatmap
├── docs/
│   ├── relazione.md               # Documentazione tecnica completa (leggere prima)
│   └── FOODRISK_Finale_aggiornata.docx   # Versione Word della relazione
└── results/                       # Output CSV e grafici
```

---

## Requisiti

- Python 3.9+
- Opzionali: SWI-Prolog (`swipl` nel PATH o variabile `SWIPL`; esecuzione reale della KB, anche senza `pyswip`) e `scikit-posthocs` (Nemenyi). Senza di essi la pipeline usa dei fallback (vedi sotto). L'ontologia non richiede librerie esterne.

```bash
pip install -r requirements.txt
```

---

## Esecuzione

```bash
# Pipeline completa (12 modelli)
python main.py

# Dataset nuovi e perturbati; coerenza KB <-> ontologia
python robustness_check.py
python kb/verify_prolog.py

# Solo generazione dataset
python data/generate_dataset.py

# Solo costruzione ontologia
python ontology/build_ontology.py
```

I risultati vengono salvati in `results/`:
- `results.csv` — metriche per modello (mean ± std su 10 run)
- `metrics_barplot.png` — F1/Precision/Recall per modello
- `f1_boxplot.png` — distribuzione F1 su 10 run
- `nemenyi_heatmap.png` — p-value post-hoc
- `owl_coverage.png` — copertura concetti OWL per classe
- `em_convergence_run0.png` — convergenza EM
- `f1_per_run.csv`, `kb_per_class.csv`, `selected_hyperparameters.csv`, `independent_check.csv`

---

## Risultati principali (`python main.py`, 10 run)

Valutazione su 10 run StratifiedShuffleSplit (test_size=30%), iperparametri scelti per CV dentro ogni run. Tabella generata da `results/results.csv`:

| Modello | Feature | F1 macro | G-Mean |
|---|---|---|---|
| Decision Tree | raw | 0.570 ± 0.035 | 0.547 ± 0.039 |
| Random Forest | raw | 0.667 ± 0.022 | 0.647 ± 0.024 |
| Logistic Regression | raw | 0.691 ± 0.017 | 0.673 ± 0.021 |
| SVM (RBF) | raw | 0.692 ± 0.015 | 0.674 ± 0.017 |
| Decision Tree | +OWL | 0.569 ± 0.036 | 0.546 ± 0.041 |
| Random Forest | +OWL | 0.661 ± 0.019 | 0.642 ± 0.022 |
| Logistic Regression | +OWL | 0.689 ± 0.015 | 0.673 ± 0.017 |
| SVM (RBF) | +OWL | 0.652 ± 0.014 | 0.634 ± 0.015 |
| BNB (da zero) | OWL bin. | 0.565 ± 0.032 | 0.544 ± 0.035 |
| BNB-EM (semi-sup.) | OWL bin. | 0.557 ± 0.031 | 0.532 ± 0.035 |
| Rete Bayesiana | OWL bin. | 0.568 ± 0.032 | 0.547 ± 0.036 |
| KB Prolog (nessun apprendimento) | raw | 0.633 ± 0.022 | 0.610 ± 0.026 |

Test di Friedman su 12 modelli: χ²=94.92, p=1.8×10⁻¹⁵ → differenze significative. Il post-hoc di Nemenyi separa LR/SVM da Decision Tree e modelli bayesiani; **nessun confronto raw vs +OWL è significativo**.

**Nota onesta sui risultati:** le feature OWL non migliorano nessun modello (DT 0.570→0.569, RF 0.667→0.661, LR 0.691→0.689, SVM 0.692→0.652). I concetti BK sono funzioni deterministiche delle feature raw, quindi modelli con capacità sufficiente li apprendono da soli. Il valore della rappresentazione ontologica sta nell'**interpretabilità**, nell'abilitare modelli simbolico-probabilistici (BNB, Rete Bayesiana con posteriori e analisi di sensibilità) e nell'avere **un'unica fonte di conoscenza** (l'ontologia) condivisa con la KB Prolog. La KB Prolog, senza apprendere, raggiunge F1 0.633; su dataset con più rumore scende al livello dei modelli bayesiani (`results/independent_check.csv`). Discussione completa in `docs/relazione.md`, sezione 9.

---

## Documentazione e verifiche

- [`docs/relazione.md`](docs/relazione.md) (e versione Word `docs/FOODRISK.docx`): documentazione tecnica completa.
- `python main.py` verifica a ogni avvio che `kb/food_kb.pl`, il riferimento Python e le soglie dell'ontologia coincidano (soglie, pesi, priorità, `compatible/2`).
- `python robustness_check.py` → `results/independent_check.csv`: 5 dataset nuovi dello stesso generatore e 5 con rumore ×1.3.
- `results/selected_hyperparameters.csv`: iperparametri scelti dalla CV in ogni run.

## Limiti (sintesi)

Ogni numero della documentazione è prodotto da uno script del repository. Elenco completo in `docs/relazione.md`, sezione 10:

- La KB è stata eseguita con SWI-Prolog 10.1.18 su tutti i 900 pasti (`python kb/verify_prolog.py`): accordo 1.0000 con il riferimento Python per `profile/2`, `compatible/2`, `risk_level/2`, `recommend_restriction/2`, `anomaly/2`. Senza `swipl` installato la pipeline usa il riferimento Python. Il controllo dinamico è stato rieseguito anche su Windows con SWI-Prolog 10.0.2 (accordo 1.0000); i percorsi passati a `consult/1` sono normalizzati con `/`.
- `food_ontology.owl` è RDF/XML scritto dal progetto: non è stato aperto con Protégé né con un reasoner OWL; l'appartenenza ai concetti BK è calcolata da un valutatore Python che legge le restrizioni dal file.
- Senza `scikit-posthocs` il Nemenyi usa un'implementazione scipy equivalente.
- Dataset sintetico: i risultati misurano la coerenza interna del sistema, non la generalizzazione a pasti reali (il test con rumore ×1.3 attenua ma non elimina il problema).
