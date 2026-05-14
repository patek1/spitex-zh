# scripts/run_clustering.py
# Clustering-Pipeline für die Spitex Standortplanungs-App.
#
# Ablauf:
#   1) grid_features.geojson laden
#   2) Features vorbereiten (Median-Impute + StandardScaler)
#   3) K-Means für k = 3, 4, 5, 6 testen → Silhouette-Scores vergleichen
#   4) Finales Clustering mit k=4
#   5) Ergebnisse speichern (GeoJSON + CSVs)
#   6) Cluster-Profile berechnen und speichern
#
# Aufruf (vom Projekt-Root):
#     python scripts/run_clustering.py

import sys
sys.path.append(".")

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.clustering import (
    CLUSTERING_FEATURES,
    DEFAULT_K,
    add_cluster_labels,
    build_cluster_profiles,
    prepare_features,
    run_kmeans_grid,
)

# Dateipfade
FILE_GRID_FEATURES = "data/processed/grid_features.geojson"

OUT_FEATURE_SUMMARY  = "data/processed/feature_summary.csv"
OUT_CORRELATIONS     = "data/processed/feature_correlations.csv"
OUT_HEATMAP          = "data/processed/feature_correlation_heatmap.png"
OUT_CLUSTERING_SCORES = "data/processed/clustering_scores.csv"
OUT_CLUSTERED_GEOJSON = "data/processed/grid_features_clustered.geojson"
OUT_CLUSTER_PROFILES  = "data/processed/cluster_profiles.csv"


def main() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1) Daten laden
    # ----------------------------------------------------------
    print("Schritt 1: grid_features.geojson laden")
    gdf = gpd.read_file(FILE_GRID_FEATURES)
    print(f"  {len(gdf)} Rasterzellen, {len(gdf.columns)} Spalten")

    # ----------------------------------------------------------
    # 2) Explorative Feature-Analyse
    # ----------------------------------------------------------
    print("Schritt 2: Deskriptive Statistik + Korrelations-Heatmap")

    feature_df = gdf[CLUSTERING_FEATURES].copy()

    # Deskriptive Statistik speichern
    summary = feature_df.describe().T.round(4)
    summary.to_csv(OUT_FEATURE_SUMMARY)

    # Korrelationsmatrix berechnen und als Heatmap speichern
    corr_matrix = feature_df.corr().round(4)
    corr_matrix.to_csv(OUT_CORRELATIONS)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax.set_title("Korrelationsmatrix der Clustering-Features", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_HEATMAP, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap gespeichert: {OUT_HEATMAP}")

    # ----------------------------------------------------------
    # 3) Features vorbereiten (Median-Impute + StandardScaler)
    # ----------------------------------------------------------
    print("Schritt 3: Features vorbereiten")
    X_scaled, used_features = prepare_features(gdf)

    # ----------------------------------------------------------
    # 4) K-Means Grid Search (k = 3, 4, 5, 6)
    # ----------------------------------------------------------
    print("Schritt 4: K-Means Grid Search")
    scores_df = run_kmeans_grid(X_scaled, k_values=[3, 4, 5, 6])
    scores_df.to_csv(OUT_CLUSTERING_SCORES, index=False)
    print(scores_df.to_string(index=False))

    # ----------------------------------------------------------
    # 5) Finales Clustering mit k=4
    # ----------------------------------------------------------
    print(f"Schritt 5: Finales Clustering mit k={DEFAULT_K}")
    gdf_clustered = add_cluster_labels(gdf, X_scaled, k=DEFAULT_K)

    # In WGS84 reprojizieren (Folium erwartet Lat/Lon)
    gdf_wgs84 = gdf_clustered.to_crs("EPSG:4326")

    out = Path(OUT_CLUSTERED_GEOJSON)
    if out.exists():
        out.unlink()
    gdf_wgs84.to_file(OUT_CLUSTERED_GEOJSON, driver="GeoJSON")
    print(f"  GeoJSON gespeichert: {OUT_CLUSTERED_GEOJSON}")

    # ----------------------------------------------------------
    # 6) Cluster-Profile berechnen und speichern
    # ----------------------------------------------------------
    print("Schritt 6: Cluster-Profile berechnen")
    profiles = build_cluster_profiles(gdf_clustered, used_features)
    profiles.to_csv(OUT_CLUSTER_PROFILES, index=False)
    print(f"  Profile gespeichert: {OUT_CLUSTER_PROFILES}")

    counts = gdf_clustered["cluster"].value_counts().sort_index()
    for cid, n in counts.items():
        print(f"  Cluster {cid}: {n} Zellen")

    print("Nächster Schritt: streamlit run app.py")


if __name__ == "__main__":
    main()
