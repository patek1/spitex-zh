# src/clustering.py
# K-Means Clustering und Cluster-Benennung für die Spitex-App.
#
# Wird genutzt von:
#   - scripts/run_clustering.py  (schreibt alle Outputs)
#   - app.py / ui_tabs.py        (cluster_name anhängen)

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Features, die ins K-Means Clustering fliessen.
# Bewusst hartkodiert – so ist klar, welche Daten ins Modell fliessen.
CLUSTERING_FEATURES = [
    "share_60plus",           # Anteil 60+ (Nachfrage-Indikator)
    "share_80plus",           # Anteil 80+ (starker Spitex-Bedarf)
    "share_einpersonen",      # Anteil Einpersonenhaushalte (Pflege-Risiko)
    "growth_10y",             # Bevölkerungswachstum 2025–2035
    "pop_total",              # Bevölkerung total (Quartiergrösse)
    "dist_nearest_spitex_m",  # Distanz zur nächsten Spitex
    "n_spitex_1000m",         # Anzahl Spitex im 1-km-Umkreis
    "dist_nearest_oev_m",     # Distanz zur nächsten ÖV-Haltestelle
    "n_oev_stops_500m",       # Anzahl ÖV-Stops im 500-m-Umkreis
]

DEFAULT_K = 4       # Anzahl Cluster – k=4 liefert die besten Silhouette-Scores
RANDOM_STATE = 42   # Zufallsstart für Reproduzierbarkeit

# Feste Farben pro Cluster (konsistent in Karte, Legende und Tab)
CLUSTER_COLORS = {
    0: "#3498db",  # Blau
    1: "#e74c3c",  # Rot
    2: "#2ecc71",  # Grün
    3: "#f39c12",  # Orange
    4: "#9b59b6",  # Lila
    5: "#1abc9c",  # Türkis
}

# -------------------------------------------------------------------
# Hardcodierte Cluster-Namen
# -------------------------------------------------------------------
# Basierend auf den Cluster-Profilen (cluster_profiles.csv) manuell abgeleitet.
# ACHTUNG: Bei erneutem Clustering bitte Profile prüfen und Mapping anpassen!
CLUSTER_NAMES: dict[int, dict[str, str]] = {
    0: {
        "name": "Wachstumsgebiet mit niedrigem Pflegeanteil",
        "description": (
            "Dieser Gebietstyp weist das höchste Bevölkerungswachstum der nächsten 10 Jahre auf "
            "(+95 % über Stadtdurchschnitt) und hat gleichzeitig einen vergleichsweise niedrigen "
            "Anteil älterer Bevölkerung. Die Versorgung mit Spitex-Standorten ist aktuell moderat. "
            "Relevant für zukunftsorientierte Standortplanung."
        ),
    },
    1: {
        "name": "Unterversorgtes Randgebiet",
        "description": (
            "Dieser Gebietstyp ist räumlich am stärksten von bestehenden Spitex-Standorten entfernt "
            "(Ø 1568 m, +48 % über Stadtdurchschnitt). Zudem sind kaum Spitex-Angebote im 1-km-Umkreis "
            "vorhanden und die ÖV-Erreichbarkeit ist eingeschränkt. "
            "Potenziell relevantes Gebiet für Versorgungslücken-Analyse."
        ),
    },
    2: {
        "name": "Älteres Wohngebiet mit Versorgungsgap",
        "description": (
            "Dieser Gebietstyp hat den höchsten Anteil älterer Bevölkerung (80+: +50 % über Stadtdurchschnitt, "
            "60+: +26 % über Stadtdurchschnitt) und liegt mit seiner Spitex-Distanz über dem Stadtmittel. "
            "Hoher latenter Pflegebedarf bei moderater Versorgungssituation."
        ),
    },
    3: {
        "name": "Zentrales gut erschlossenes Gebiet",
        "description": (
            "Dieser Gebietstyp ist am besten mit ÖV erschlossen (ÖV-Stops im 500m-Umkreis: +52 % über Ø) "
            "und hat die höchste Spitex-Dichte (n_spitex_1000m: +79 % über Ø). "
            "Der Anteil an Einpersonenhaushalten ist hoch. "
            "Aktuell gut versorgt – weniger dringlicher Handlungsbedarf."
        ),
    },
}


def attach_cluster_name(df: pd.DataFrame) -> pd.DataFrame:
    # Cluster-Namen-Spalte aus dem hardcodierten Mapping hinzufügen
    name_map = {cid: info["name"] for cid, info in CLUSTER_NAMES.items()}
    result = df.copy()
    result["cluster_name"] = result["cluster"].apply(
        lambda cid: name_map.get(int(cid), f"Cluster {cid}") if pd.notna(cid) else None
    )
    return result


def prepare_features(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, list[str]]:
    # Feature-Matrix aus den definierten Clustering-Features extrahieren
    X = gdf[CLUSTERING_FEATURES].values

    # Fehlende Werte mit Median auffüllen (robust gegenüber Ausreissern)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Standardisieren (Z-Score) – wichtig, da K-Means distanzbasiert ist
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, CLUSTERING_FEATURES


def run_kmeans_grid(X_scaled: np.ndarray, k_values: list[int] = None) -> pd.DataFrame:
    # K-Means für verschiedene k-Werte testen und Silhouette-Scores vergleichen
    if k_values is None:
        k_values = [3, 4, 5, 6]

    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "silhouette_score": round(sil, 4), "inertia": round(km.inertia_, 2)})

    return pd.DataFrame(results).sort_values("silhouette_score", ascending=False)


def add_cluster_labels(gdf: gpd.GeoDataFrame, X_scaled: np.ndarray, k: int = DEFAULT_K) -> gpd.GeoDataFrame:
    # Finales K-Means-Clustering mit k Clustern durchführen
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X_scaled)

    result = gdf.copy()
    result["cluster"] = labels.astype(int)
    return result


def build_cluster_profiles(gdf: gpd.GeoDataFrame, used_features: list[str]) -> pd.DataFrame:
    # Mittelwert pro Feature je Cluster berechnen
    profile = gdf.groupby("cluster")[used_features].mean().round(4)

    # Anzahl Rasterzellen pro Cluster
    counts = gdf.groupby("cluster").size().rename("n_zellen")
    profile = pd.concat([counts, profile], axis=1).reset_index()
    return profile
