"""
src/supervised.py
Modelli supervisionati: Decision Tree, Random Forest, Logistic Regression, SVM.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from config import DT_PARAMS, RF_PARAMS, LR_PARAMS, SVM_PARAMS, TUNING_GRIDS


def get_models():
    """Restituisce un dict {nome: modello} con tutti i classificatori supervisionati."""
    return {
        "DecisionTree":       DecisionTreeClassifier(**DT_PARAMS),
        "RandomForest":       RandomForestClassifier(**RF_PARAMS),
        "LogisticRegression": LogisticRegression(**LR_PARAMS),
        "SVM_RBF":            SVC(**SVM_PARAMS),
    }


def train_predict(model, X_train, y_train, X_test):
    """Addestra il modello e restituisce le predizioni sul test set."""
    model.fit(X_train, y_train)
    return model.predict(X_test)


def tuned_fit(name, X_train, y_train, seed=0):
    """
    Scelta degli iperparametri per cross-validation a 5 fold (F1 macro) SUL SOLO
    training set ricevuto (nessun accesso al test set), poi refit del modello
    completo con i parametri scelti. La griglia e' TUNING_GRIDS[name] (config.py).
    Per velocita' la CV usa stimatori alleggeriti (RF con 40 alberi, SVM senza
    probabilita'); il modello finale usa i parametri pieni di config.py.
    Restituisce (modello addestrato, dict dei parametri scelti).
    """
    est = get_models()[name]
    if name == "SVM_RBF":
        est.set_params(probability=False)
    if name == "RandomForest":
        est.set_params(n_estimators=40)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(est, TUNING_GRIDS[name], cv=cv, scoring="f1_macro", n_jobs=1, refit=False)
    gs.fit(X_train, y_train)
    best = gs.best_params_
    final = get_models()[name].set_params(**best)
    final.fit(X_train, y_train)
    return final, best
