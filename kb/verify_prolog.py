"""
kb/verify_prolog.py
Verifica che KB Prolog (food_kb.pl), fallback Python e ontologia siano coerenti.

1) CONTROLLO STRUTTURALE (sempre eseguito, nessuna dipendenza): legge food_kb.pl
   come testo e confronta con le regole Python (src/prolog_kb.py) e con le
   soglie dell'ontologia (src/ontology_features.LEVELS):
     - soglie di discretizzazione low/medium/high di ogni feature
     - pesi e penalita' di score/3 per ogni profilo
     - ordine di priorita' priority_order/1
     - congiunzioni di compatible/2
2) CONTROLLO DINAMICO (solo se SWI-Prolog e' disponibile, via pyswip o `swipl` nel PATH):
   esegue la KB Prolog su tutto il dataset e misura l'accordo con il riferimento Python
   per profile/2, compatible/2, risk_level/2, recommend_restriction/2 e anomaly/2.

Uso:  python kb/verify_prolog.py
"""

import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PROLOG_KB_PATH, RAW_FEATURES, DATASET_PATH
from src.ontology_features import LEVELS
from src import prolog_kb as pk


def _parse_levels(text):
    """attr -> {'low': t_low, 'high': t_high} dalle clausole di discretizzazione."""
    found = {}
    pat = re.compile(r"(\w+?)(?:_level)?\(Meal, (low|medium|high)\)\s*:-\s*meal_\w+\(Meal, V\)((?:[^.\n]|\.(?=\d))*)\.[ \t]*\n")
    for attr, lvl, rest in pat.findall(text):
        if attr not in pk.FEATURE_OF:
            continue
        cmps = re.findall(r"V\s*(<|>=)\s*([\d.]+)", rest)
        d = found.setdefault(attr, {})
        if lvl == "low":
            d["low"] = float([v for o, v in cmps if o == "<"][0])
        elif lvl == "high":
            d["high"] = float([v for o, v in cmps if o == ">="][0])
        else:
            d["m_from"] = float([v for o, v in cmps if o == ">="][0])
            d["m_to"] = float([v for o, v in cmps if o == "<"][0])
    return found


def _parse_scores(text):
    rules = {}
    for prof, body in re.findall(r"score\(Meal, (\w+), S\) :-(.*?)\n\n", text, re.S):
        lists = {v: (re.match(r"(\w+?)(?:_level)?$", a).group(1), l)
                 for a, l, v in [(m[0], m[1], m[2]) for m in
                                 re.findall(r"findall\(1, (\w+)\(Meal, (\w+)\), (\w+)\)", body)]}
        letter = dict((lst, var) for lst, var in re.findall(r"length\((\w+), (\w+)\)", body))
        var_attr = {letter[v]: lists[v] for v in lists}
        expr = re.search(r"S is (.*)\.", body).group(1).replace("(", "").replace(")", "")
        base, terms = 0, []
        for sign, tok in re.findall(r"([+-]?)\s*([A-Za-z0-9*]+)", expr.replace(" ", "")):
            sgn = -1 if sign == "-" else 1
            m = re.fullmatch(r"([A-Z]\w*)(?:\*(\d+))?", tok)
            if m:
                a, l = var_attr[m.group(1)]
                terms.append((a, l, sgn * int(m.group(2) or 1)))
            else:
                base += sgn * int(tok)
        rules[prof] = (base, terms)
    return rules


def _parse_compatible(text):
    out = {}
    for cond, body in re.findall(r"compatible\(Meal, (\w+)\) :-(.*?)\.\n", text, re.S):
        items = []
        for neg, attr, lvl in re.findall(r"(\\\+\s*)?(\w+?)(?:_level)?\(Meal, (\w+)\)", body):
            items.append((attr, lvl, bool(neg)))
        out[cond] = items
    return out


def check_static(verbose=False):
    text = open(PROLOG_KB_PATH, encoding="utf8").read()
    problems = []

    # 1) soglie
    parsed = _parse_levels(text)
    for attr, feat in pk.FEATURE_OF.items():
        low, high = LEVELS[feat]
        d = parsed.get(attr, {})
        exp = {"low": low, "high": high, "m_from": low, "m_to": high}
        for k, v in exp.items():
            if abs(d.get(k, -1) - v) > 1e-9:
                problems.append(f"soglia {attr}.{k}: Prolog={d.get(k)} ontologia={v}")

    # 2) punteggi
    ps = _parse_scores(text)
    for prof, (base, rules) in pk.SCORE_RULES.items():
        pbase, prules = ps.get(prof, (None, []))
        if pbase != base or sorted(prules) != sorted(rules):
            problems.append(f"score({prof}): Prolog={(pbase, sorted(prules))} Python={(base, sorted(rules))}")

    # 3) priorita'
    m = re.search(r"priority_order\(\[(.*?)\]\)", text)
    order = [x.strip() for x in m.group(1).split(",")]
    if order != pk.PRIORITY:
        problems.append(f"priority_order: Prolog={order} Python={pk.PRIORITY}")

    # 4) compatibilita'
    pc = _parse_compatible(text)
    for cond, rules in pk.COMPATIBLE_RULES.items():
        if sorted(pc.get(cond, [])) != sorted(rules):
            problems.append(f"compatible({cond}): Prolog={pc.get(cond)} Python={rules}")

    if verbose:
        print(f"  soglie verificate:   {len(pk.FEATURE_OF)} attributi x 4 valori")
        print(f"  profili verificati:  {len(pk.SCORE_RULES)} (pesi, penalita', costanti)")
        print(f"  compatible/2:        {len(pk.COMPATIBLE_RULES)} condizioni; priorita': {order}")
    return problems


def _levels(row):
    return {a: pk.level_of(a, row[f]) for a, f in pk.FEATURE_OF.items()}


def _py_reference(row):
    """Riferimento Python di profile/2, compatible/2, risk_level/2, recommend_restriction/2, anomaly/2."""
    lv = _levels(row); prof = pk._infer_profile_python(row)
    comp = {c: pk._is_compatible_python(row, c) for c in pk.COMPATIBLE_RULES}
    nh = sum(lv[a] == "high" for a in ("fat", "sodium", "sugar"))
    risk = "high" if nh == 3 else ("moderate" if nh >= 1 else "low")
    rec = set()
    if lv["sodium"] == "high": rec.add("reduce_sodium")
    if lv["sugar"] == "high": rec.add("reduce_sugar")
    if lv["fat"] == "high": rec.add("reduce_saturated_fat")
    if lv["fiber"] == "low" and prof != pk.PROFILE_ID["sportivo"]: rec.add("increase_fiber")
    if not rec: rec.add("none")
    anom = set()
    if prof == pk.PROFILE_ID["diabete"] and lv["sugar"] == "high": anom.add("diabete_high_sugar")
    if prof == pk.PROFILE_ID["sportivo"] and lv["protein"] == "low": anom.add("sportivo_low_protein")
    return prof, comp, risk, rec, anom


def check_dynamic():
    """Esegue food_kb.pl con SWI-Prolog (pyswip o sottoprocesso) e confronta con il riferimento Python."""
    import numpy as np, pandas as pd
    df = pd.read_csv(DATASET_PATH)
    X = df[RAW_FEATURES].values
    if not (pk.PROLOG_AVAILABLE or pk.SWIPL_PATH):
        print("  [dinamico] SWI-Prolog non disponibile: confronto di esecuzione NON eseguito.")
        print("             Installare SWI-Prolog (swipl nel PATH o variabile SWIPL), poi rilanciare.")
        return None
    if not pk.SWIPL_PATH:       # solo pyswip: confronto sul solo profile/2
        kb = pk.PrologKB(); pred_pl = kb.infer_profile(X)
        pred_py = np.array([pk._infer_profile_python(dict(zip(RAW_FEATURES, r))) for r in X])
        agree = float((pred_pl == pred_py).mean())
        print(f"  [dinamico] profile/2, accordo Prolog vs Python su {len(X)} pasti: {agree:.4f}")
        return agree
    ref = [_py_reference(dict(zip(RAW_FEATURES, r))) for r in X]
    kb_ = pk.KB_PATH_FOR_TESTS if hasattr(pk, "KB_PATH_FOR_TESTS") else PROLOG_KB_PATH
    res = {}
    res["profile/2"] = [pk.PROFILE_ID.get(r, 0) == ref[i][0]
                        for i, r in enumerate(pk.swipl_batch(kb_, X, "profile({M}, R)"))]
    for c in pk.COMPATIBLE_RULES:
        out = pk.swipl_batch(kb_, X, f"( compatible({{M}}, {c}) -> R = yes ; R = no )")
        res[f"compatible/2 ({c})"] = [(o == "yes") == ref[i][1][c] for i, o in enumerate(out)]
    res["risk_level/2"] = [o == ref[i][2] for i, o in enumerate(pk.swipl_batch(kb_, X, "risk_level({M}, R)"))]
    rec = pk.swipl_batch(kb_, X, "findall(T, recommend_restriction({M}, T), L), sort(L, S), atomic_list_concat(S, '+', R)")
    res["recommend_restriction/2"] = [set(o.split("+")) == ref[i][3] for i, o in enumerate(rec)]
    an = pk.swipl_batch(kb_, X, "findall(T, anomaly({M}, T), L), sort(L, S), atomic_list_concat(S, '+', R0), (R0 == '' -> R = none ; R = R0)")
    res["anomaly/2"] = [(set() if o == "none" else set(o.split("+"))) == ref[i][4] for i, o in enumerate(an)]
    worst = 1.0
    for k, v in res.items():
        acc = float(np.mean(v)); worst = min(worst, acc)
        print(f"  [dinamico] {k:34s} accordo Prolog vs Python su {len(X)} pasti: {acc:.4f}")
    return worst


def main():
    print("Verifica coerenza KB Prolog <-> fallback Python <-> ontologia")
    problems = check_static(verbose=True)
    if problems:
        print("  DISCREPANZE:"); [print("   -", p) for p in problems]; sys.exit(1)
    print("  [statico] OK: soglie, pesi, priorita' e compatible/2 coincidono.")
    check_dynamic()


if __name__ == "__main__":
    main()
