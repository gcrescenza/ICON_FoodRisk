"""
ontology/spec.py
Specifica dichiarativa dell'ontologia FOODRISK: UNICA fonte di verita' per
tassonomia, proprieta', individui, livelli nutrizionali (low/high) e definizione
dei 12 concetti di Background Knowledge.

Da questa specifica `owl_io.write_owl` genera `food_ontology.owl`; la pipeline
(`src/ontology_features.py`, `src/prolog_kb.py`) legge poi soglie e concetti
DAL FILE OWL, non da costanti Python duplicate.
"""

NS = "http://foodrisk.org/ontology/food"

# (classe, superclasse) in ordine di dichiarazione
CLASS_TREE = [
    ("Meal", None), ("Nutrient", None), ("HealthCondition", None),
    ("IngredientCategory", None), ("RiskLevel", None), ("MealType", None),
    ("Macronutrient", "Nutrient"), ("Micronutrient", "Nutrient"),
    ("Carbohydrate", "Macronutrient"), ("Protein", "Macronutrient"), ("Fat", "Macronutrient"),
    ("SaturatedFat", "Fat"), ("UnsaturatedFat", "Fat"),
    ("Fiber", "Carbohydrate"), ("Sugar", "Carbohydrate"),
    ("Sodium", "Micronutrient"), ("Potassium", "Micronutrient"),
    ("ProcessedIngredient", "IngredientCategory"), ("WholeFoodIngredient", "IngredientCategory"),
    ("HighSodiumIngredient", "ProcessedIngredient"), ("HighSugarIngredient", "ProcessedIngredient"),
    ("HighFiberIngredient", "WholeFoodIngredient"), ("LeanProteinIngredient", "WholeFoodIngredient"),
    ("Diabetes", "HealthCondition"), ("Hypertension", "HealthCondition"),
    ("AthleticPerformance", "HealthCondition"), ("GeneralWellness", "HealthCondition"),
    ("MetabolicRisk", "HealthCondition"),
    ("LowRisk", "RiskLevel"), ("ModerateRisk", "RiskLevel"), ("HighRisk", "RiskLevel"),
    ("Breakfast", "MealType"), ("Lunch", "MealType"), ("Dinner", "MealType"), ("Snack", "MealType"),
]

# (proprieta', dominio, range)
OBJECT_PROPS = [
    ("containsNutrient", "Meal", "Nutrient"),
    ("hasIngredientCategory", "Meal", "IngredientCategory"),
    ("hasMealType", "Meal", "MealType"),
    ("suitableFor", "Meal", "HealthCondition"),
    ("contraindicatedFor", "Meal", "HealthCondition"),
    ("hasRiskLevel", "Meal", "RiskLevel"),
    ("requiresLowLevelOf", "HealthCondition", "Nutrient"),    # BK
    ("requiresHighLevelOf", "HealthCondition", "Nutrient"),   # BK
]

# feature raw -> (data property del Meal, tipo XSD)
FEATURE_PROPS = {
    "calories":                ("hasCalories", "float"),
    "sodium_mg":               ("hasSodium", "float"),
    "sugar_g":                 ("hasSugar", "float"),
    "saturated_fat_g":         ("hasSaturatedFat", "float"),
    "fiber_g":                 ("hasFiber", "float"),
    "protein_g":               ("hasProtein", "float"),
    "glycemic_index":          ("hasGlycemicIndex", "float"),
    "n_processed_ingredients": ("hasProcessedIngredientCount", "integer"),
    "potassium_mg":            ("hasPotassium", "float"),
}

# Livelli nutrizionali: feature -> (lowBelow, highFrom)
#   low    = valore <  lowBelow
#   high   = valore >= highFrom
#   medium = tra i due
# Per il conteggio degli ingredienti processati le soglie sono in unita' di
# conteggio (low = 0-1 ingredienti, high = 3 o piu').
LEVELS = {
    "glycemic_index":          (0.38, 0.58),
    "sugar_g":                 (0.28, 0.50),
    "sodium_mg":               (0.30, 0.55),
    "fiber_g":                 (0.38, 0.58),
    "protein_g":               (0.42, 0.65),
    "saturated_fat_g":         (0.30, 0.55),
    "potassium_mg":            (0.42, 0.60),
    "calories":                (0.42, 0.60),
    "n_processed_ingredients": (2, 3),
}

# Predicati di livello usabili nei concetti
#   low -> < lowBelow ; notLow -> >= lowBelow ; notHigh -> < highFrom ; high -> >= highFrom
BK_CONCEPTS = {
    "bk_diabetes_compatible":     [("glycemic_index", "low"), ("sugar_g", "low"), ("fiber_g", "high")],
    "bk_diabetes_risk":           [("glycemic_index", "high"), ("sugar_g", "high")],
    "bk_hypertension_compatible": [("sodium_mg", "low"), ("potassium_mg", "notLow")],
    "bk_hypertension_risk":       [("sodium_mg", "high"), ("potassium_mg", "low")],
    "bk_athletic_compatible":     [("protein_g", "high"), ("saturated_fat_g", "low"), ("calories", "high")],
    "bk_metabolic_risk":          [("saturated_fat_g", "high"), ("sodium_mg", "high"), ("sugar_g", "high")],
    "bk_ultra_processed_heavy":   [("n_processed_ingredients", "high"), ("saturated_fat_g", "notLow")],
    "bk_balanced_macros":         [("sugar_g", "notHigh"), ("saturated_fat_g", "notHigh"),
                                   ("sodium_mg", "notHigh"), ("protein_g", "notLow")],
    "bk_low_glycemic_load":       [("glycemic_index", "notHigh"), ("fiber_g", "notLow")],
    "bk_heart_healthy":           [("saturated_fat_g", "low"), ("sodium_mg", "notHigh"), ("potassium_mg", "notLow")],
    "bk_high_density_protein":    [("protein_g", "high"), ("calories", "notHigh")],
    "bk_whole_food_profile":      [("n_processed_ingredients", "low"), ("fiber_g", "notLow")],
}

# Individui: (nome, classe, {proprieta': [individui target]})
INDIVIDUALS = [
    ("sugar_constraint", "Sugar", {}), ("carb_constraint", "Carbohydrate", {}),
    ("fiber_constraint", "Fiber", {}), ("sodium_constraint", "Sodium", {}),
    ("potassium_constraint", "Potassium", {}), ("protein_constraint", "Protein", {}),
    ("satfat_constraint", "SaturatedFat", {}), ("satfat_constraint2", "SaturatedFat", {}),
    ("sodium_constraint2", "Sodium", {}), ("sugar_constraint2", "Sugar", {}),
    ("diabetes_profile", "Diabetes",
        {"requiresLowLevelOf": ["sugar_constraint", "carb_constraint"],
         "requiresHighLevelOf": ["fiber_constraint"]}),
    ("hypertension_profile", "Hypertension",
        {"requiresLowLevelOf": ["sodium_constraint"],
         "requiresHighLevelOf": ["potassium_constraint"]}),
    ("athletic_profile", "AthleticPerformance",
        {"requiresHighLevelOf": ["protein_constraint"],
         "requiresLowLevelOf": ["satfat_constraint"]}),
    ("metabolic_risk_profile", "MetabolicRisk",
        {"requiresLowLevelOf": ["satfat_constraint2", "sodium_constraint2", "sugar_constraint2"]}),
]
