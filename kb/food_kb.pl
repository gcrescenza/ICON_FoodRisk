% ─────────────────────────────────────────────────────────────────────────────
% kb/food_kb.pl  —  Knowledge Base Prolog per FOODRISK
% Regole per: classificazione profilo, compatibilità pasto-condizione clinica,
%             stima livello di rischio, raccomandazione restrizioni.
%
% NOTA SULLA CALIBRAZIONE: le soglie di discretizzazione sono state allineate
% alle mediane per classe osservate nel dataset generato da
% data/generate_dataset.py (vedi docs/relazione.md, sez. 3.3) per evitare che
% regole congiuntive troppo rigide collassino quasi sempre sul profilo
% di default in presenza del realistico rumore intra-classe introdotto nel
% dataset. La logica di profile/2 usa un meccanismo di voto pesato
% (conteggio dei vincoli soddisfatti per profilo) anziché una congiunzione
% rigida di tutti i vincoli, per essere più robusta al rumore pur restando
% un ragionamento simbolico esplicito e non un semplice pattern matching.
% ─────────────────────────────────────────────────────────────────────────────

% ── Discretizzazione livelli (valori normalizzati 0-1) ────────────────────────
% Soglie calibrate sulle mediane osservate per classe nel dataset sintetico.

% ── Punteggio per profilo: conta quanti vincoli caratteristici sono rispettati ─

score(Meal, diabete, S) :-
    findall(1, sugar_level(Meal, low), L1),
    findall(1, fiber_level(Meal, high), L2),
    findall(1, glycemic(Meal, low), L3),
    findall(1, glycemic(Meal, medium), L4),
    findall(1, sodium_level(Meal, high), Pen1),
    findall(1, calorie_level(Meal, high), Pen2),
    length(L1, A), length(L2, B), length(L3, C), length(L4, D),
    length(Pen1, P1), length(Pen2, P2),
    S is (A*2) + (B*2) + (C*2) + D - P1 - P2.

score(Meal, iperteso, S) :-
    findall(1, sodium_level(Meal, low), L1),
    findall(1, potassium_level(Meal, high), L2),
    findall(1, potassium_level(Meal, medium), L3),
    findall(1, protein_level(Meal, high), Pen1),
    length(L1, A), length(L2, B), length(L3, C), length(Pen1, P1),
    S is (A*2) + (B*2) + C - P1.

score(Meal, sportivo, S) :-
    findall(1, protein_level(Meal, high), L1),
    findall(1, fat_level(Meal, low), L2),
    findall(1, calorie_level(Meal, high), L3),
    length(L1, A), length(L2, B), length(L3, C),
    S is (A*3) + B + C.

score(Meal, ipercalorico, S) :-
    findall(1, fat_level(Meal, high), L1),
    findall(1, sodium_level(Meal, high), L2),
    findall(1, sugar_level(Meal, high), L3),
    findall(1, calorie_level(Meal, high), L4),
    length(L1, A), length(L2, B), length(L3, C), length(L4, D),
    S is (A*2) + B + C + D.

score(Meal, bilanciato, S) :-
    findall(1, sugar_level(Meal, medium), L1),
    findall(1, sodium_level(Meal, medium), L2),
    findall(1, fat_level(Meal, low), L3),
    findall(1, glycemic(Meal, medium), L4),
    findall(1, calorie_level(Meal, high), Pen1),
    length(L1, A), length(L2, B), length(L3, C), length(L4, D), length(Pen1, P1),
    S is A + B + C + D - P1 + 1.

% ── Profilo finale: quello con punteggio massimo (con priorità clinica in caso
%    di pareggio: diabete > iperteso > sportivo > ipercalorico > bilanciato) ───

profile(Meal, Profile) :-
    score(Meal, diabete, Sd),
    score(Meal, iperteso, Si),
    score(Meal, sportivo, Ss),
    score(Meal, ipercalorico, Sk),
    score(Meal, bilanciato, Sb),
    pick_best([diabete-Sd, iperteso-Si, sportivo-Ss, ipercalorico-Sk, bilanciato-Sb],
              Profile).

pick_best(Scored, Best) :-
    max_member(comparator, _-MaxScore, Scored),
    findall(P-S, (member(P-S, Scored), S =:= MaxScore), Tied),
    priority_order(Order),
    member(B, Order),
    member(B-_, Tied), !,
    Best = B.      % unifica DOPO la scelta: con Profile gia' istanziato la priorita' resta valida

comparator(_-S1, _-S2) :- S1 =< S2.

priority_order([diabete, iperteso, sportivo, ipercalorico, bilanciato]).

% ── Discretizzazione singole feature (soglie calibrate sul dataset) ──────────
glycemic(Meal, low)    :- meal_glycemic_index(Meal, V), V < 0.38.
glycemic(Meal, medium) :- meal_glycemic_index(Meal, V), V >= 0.38, V < 0.58.
glycemic(Meal, high)   :- meal_glycemic_index(Meal, V), V >= 0.58.

sugar_level(Meal, low)    :- meal_sugar(Meal, V), V < 0.28.
sugar_level(Meal, medium) :- meal_sugar(Meal, V), V >= 0.28, V < 0.50.
sugar_level(Meal, high)   :- meal_sugar(Meal, V), V >= 0.50.

sodium_level(Meal, low)    :- meal_sodium(Meal, V), V < 0.30.
sodium_level(Meal, medium) :- meal_sodium(Meal, V), V >= 0.30, V < 0.55.
sodium_level(Meal, high)   :- meal_sodium(Meal, V), V >= 0.55.

fiber_level(Meal, low)    :- meal_fiber(Meal, V), V < 0.38.
fiber_level(Meal, medium) :- meal_fiber(Meal, V), V >= 0.38, V < 0.58.
fiber_level(Meal, high)   :- meal_fiber(Meal, V), V >= 0.58.

protein_level(Meal, low)    :- meal_protein(Meal, V), V < 0.42.
protein_level(Meal, medium) :- meal_protein(Meal, V), V >= 0.42, V < 0.65.
protein_level(Meal, high)   :- meal_protein(Meal, V), V >= 0.65.

fat_level(Meal, low)    :- meal_saturated_fat(Meal, V), V < 0.30.
fat_level(Meal, medium) :- meal_saturated_fat(Meal, V), V >= 0.30, V < 0.55.
fat_level(Meal, high)   :- meal_saturated_fat(Meal, V), V >= 0.55.

potassium_level(Meal, low)    :- meal_potassium(Meal, V), V < 0.42.
potassium_level(Meal, medium) :- meal_potassium(Meal, V), V >= 0.42, V < 0.60.
potassium_level(Meal, high)   :- meal_potassium(Meal, V), V >= 0.60.

calorie_level(Meal, low)    :- meal_calories(Meal, V), V < 0.42.
calorie_level(Meal, medium) :- meal_calories(Meal, V), V >= 0.42, V < 0.60.
calorie_level(Meal, high)   :- meal_calories(Meal, V), V >= 0.60.

% ── Compatibilità pasto ↔ condizione clinica ─────────────────────────────────
% Un pasto è compatibile con una condizione se rispetta tutti i vincoli
% nutrizionali associati a quella condizione (qui invece congiunzione rigida,
% intenzionalmente più stringente di profile/2: serve a verificare un singolo
% requisito clinico puntuale, non a classificare l'intero pasto).

compatible(Meal, diabete) :-
    sugar_level(Meal, low),
    glycemic(Meal, low).

compatible(Meal, iperteso) :-
    sodium_level(Meal, low),
    \+ potassium_level(Meal, low).

compatible(Meal, sportivo) :-
    protein_level(Meal, high),
    fat_level(Meal, low).

compatible(Meal, metabolic_risk) :-
    fat_level(Meal, low),
    sodium_level(Meal, low),
    sugar_level(Meal, low).

% ── Stima livello di rischio complessivo ──────────────────────────────────────
risk_level(Meal, high) :-
    fat_level(Meal, high),
    sodium_level(Meal, high),
    sugar_level(Meal, high).

risk_level(Meal, moderate) :-
    \+ risk_level(Meal, high),
    ( fat_level(Meal, high) ; sodium_level(Meal, high) ; sugar_level(Meal, high) ).

risk_level(Meal, low) :-
    \+ risk_level(Meal, high),
    \+ risk_level(Meal, moderate).

% ── Raccomandazione restrizioni nutrizionali ──────────────────────────────────
recommend_restriction(Meal, reduce_sodium) :-
    sodium_level(Meal, high).

recommend_restriction(Meal, reduce_sugar) :-
    sugar_level(Meal, high).

recommend_restriction(Meal, reduce_saturated_fat) :-
    fat_level(Meal, high).

recommend_restriction(Meal, increase_fiber) :-
    fiber_level(Meal, low),
    \+ profile(Meal, sportivo).

recommend_restriction(Meal, none) :-
    \+ recommend_restriction(Meal, reduce_sodium),
    \+ recommend_restriction(Meal, reduce_sugar),
    \+ recommend_restriction(Meal, reduce_saturated_fat),
    \+ recommend_restriction(Meal, increase_fiber).

% ── Regole anomalia (errore di etichettatura) ─────────────────────────────────
anomaly(Meal, diabete_high_sugar) :-
    profile(Meal, diabete),
    sugar_level(Meal, high).

anomaly(Meal, sportivo_low_protein) :-
    profile(Meal, sportivo),
    protein_level(Meal, low).

% ── Predicati placeholder (istanziati da Python) ─────────────────────────────
:- dynamic meal_calories/2.
:- dynamic meal_sodium/2.
:- dynamic meal_sugar/2.
:- dynamic meal_saturated_fat/2.
:- dynamic meal_fiber/2.
:- dynamic meal_protein/2.
:- dynamic meal_glycemic_index/2.
:- dynamic meal_potassium/2.
