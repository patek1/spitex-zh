# src/scoring.py
# Berechnet den interaktiven Empfehlungsscore für die Spitex-App.
#
# Der Score ist KEINE ML-Methode, sondern eine transparente Entscheidungshilfe,
# die als zweite Schicht über das K-Means Clustering gelegt wird.

import pandas as pd

# Vordefinierte Strategien (Preset-Gewichte für die Sidebar)
STRATEGIES: dict[str, dict[str, float]] = {
    "Ausgewogen": {
        "demand": 25, "undersupply": 25, "growth": 25, "accessibility": 25,
    },
    "Fokus Pflegebedarf": {
        "demand": 55, "undersupply": 20, "growth": 15, "accessibility": 10,
    },
    "Fokus Unterversorgung": {
        "demand": 20, "undersupply": 55, "growth": 10, "accessibility": 15,
    },
    "Fokus Wachstum": {
        "demand": 15, "undersupply": 20, "growth": 55, "accessibility": 10,
    },
    "Fokus Erreichbarkeit": {
        "demand": 15, "undersupply": 15, "growth": 15, "accessibility": 55,
    },
}

# Anzeigenamen für die 4 Score-Komponenten (Labels in Sidebar + Tabelle)
SCORE_COMPONENT_LABELS: dict[str, str] = {
    "demand": "Pflegebedarf",
    "undersupply": "Unterversorgung",
    "growth": "Wachstum",
    "accessibility": "Erreichbarkeit",
}

# Mapping: Score-Key → Spaltenname im GeoDataFrame
SCORE_COL_MAP: dict[str, str] = {
    "demand": "demand_score",
    "undersupply": "undersupply_score",
    "growth": "growth_score",
    "accessibility": "accessibility_score",
}


def compute_recommendation_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    # Gewichte normieren, damit sie zusammen 1.0 ergeben
    total = sum(weights.values())
    w = {k: v / total for k, v in weights.items()}

    # Gewichtete Summe der vier Score-Komponenten berechnen
    result = (
        w["demand"]          * df["demand_score"].fillna(0.5)
        + w["undersupply"]   * df["undersupply_score"].fillna(0.5)
        + w["growth"]        * df["growth_score"].fillna(0.5)
        + w["accessibility"] * df["accessibility_score"].fillna(0.5)
    ).clip(0.0, 1.0)

    return result
