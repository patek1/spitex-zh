# scripts/build_features.py
# Feature-Building-Pipeline für die Spitex Standortplanungs-App.
#
# Ablauf:
#   1) Geodaten laden (Quartiere, Spitex, ÖV)
#   2) Tabellen-Datensätze bereinigen (Bevölkerung, Haushalte, Szenarien)
#   3) Quartier-Feature-Tabelle bauen (Joins auf qnr)
#   4) Raster über die Stadt erzeugen (200 m Zellen)
#   5) Rasterzellen den Quartieren zuordnen (Zentroid-Join)
#   6) Punkt-basierte Features (Distanz + Anzahl Spitex / ÖV)
#   7) Score-Vorbereitungs-Features berechnen (0..1 normiert)
#   8) Output als GeoJSON speichern
#
# Aufruf (vom Projekt-Root):
#     python scripts/build_features.py

import sys
sys.path.append(".")

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from src.cleaning import DEFAULT_YEAR, clean_households, clean_population_age, clean_population_scenarios

# Konfiguration
CELL_SIZE_M = 200         # Zellgrösse in Metern
SPITEX_RADIUS_M = 1000    # Radius für Spitex-Anzahl
OEV_RADIUS_M = 500        # Radius für ÖV-Haltestellen-Anzahl

# Dateipfade
FILE_QUARTIERE = "data/raw/Statistische-Quartiere-ZH.gpkg"
FILE_SPITEX    = "data/raw/Spitexstandorte-Stadt-ZH.gpkg"
FILE_OEV       = "data/raw/Haltestellen_des_offentlichen_Verkehrs_-OGD.gpkg"
FILE_POPULATION = "data/raw/Bevölkerung-Altergruppe-Quartier-ZH.csv"
FILE_HOUSEHOLDS = "data/raw/Privathaushalte-Haushaltsform-Stadtquartier-ZH.csv"
FILE_SCENARIOS  = "data/raw/Bevölkerungsszenarien.csv"

OUT_GEOJSON = "data/processed/grid_features.geojson"


def min_max_normalize(series: pd.Series) -> pd.Series:
    # Skaliert eine Zahlenreihe linear auf 0..1
    vmin, vmax = series.min(), series.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(0.5, index=series.index)
    return (series - vmin) / (vmax - vmin)


def build_grid(boundary_gdf: gpd.GeoDataFrame, cell_size_m: int) -> gpd.GeoDataFrame:
    # Begrenzungsrahmen des Stadtgebiets bestimmen
    area = boundary_gdf.geometry.union_all()
    minx, miny, maxx, maxy = area.bounds
    print(f"  Raster-BBox: ({minx:.0f}, {miny:.0f}) – ({maxx:.0f}, {maxy:.0f}) m")

    # Gitter aus quadratischen Zellen erzeugen
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)
    xx, yy = np.meshgrid(xs, ys)
    x_flat, y_flat = xx.flatten(), yy.flatten()
    cells = [box(x, y, x + cell_size_m, y + cell_size_m) for x, y in zip(x_flat, y_flat)]

    grid = gpd.GeoDataFrame(
        {
            "geometry": cells,
            "centroid_x_m": x_flat + cell_size_m / 2,
            "centroid_y_m": y_flat + cell_size_m / 2,
        },
        crs="EPSG:2056",
    )

    # Nur Zellen behalten, die das Stadtgebiet schneiden, dann zuschneiden
    grid = grid[grid.geometry.intersects(area)].reset_index(drop=True)
    grid = gpd.clip(grid, boundary_gdf).reset_index(drop=True)
    grid["cell_id"] = range(len(grid))

    print(f"  Raster fertig: {len(grid)} Zellen.")
    return grid


def main() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1) Geodaten laden
    # ----------------------------------------------------------
    print("Schritt 1: Geodaten laden")

    quartiere = gpd.read_file(FILE_QUARTIERE, layer="stzh.adm_statistische_quartiere_v").to_crs("EPSG:2056")
    quartiere = quartiere[["qnr", "qname", "knr", "kname", "geometry"]]
    print(f"  Quartiere: {len(quartiere)}")

    spitex = gpd.read_file(FILE_SPITEX, layer="stzh.poi_spitex_view").to_crs("EPSG:2056")
    spitex = spitex[["name", "adresse", "geometry"]]
    print(f"  Spitex: {len(spitex)}")

    oev = gpd.read_file(FILE_OEV, layer="zvv_haltestellen_p").to_crs("EPSG:2056")
    oev = oev[["chstname", "symb_text", "geometry"]]
    print(f"  ÖV-Haltestellen: {len(oev)}")

    # Stadtgrenze = Union aller Quartiere
    boundary_geom = quartiere.geometry.union_all()
    boundary = gpd.GeoDataFrame({"name": ["Zürich"], "geometry": [boundary_geom]}, crs="EPSG:2056")

    # ----------------------------------------------------------
    # 2) Tabellen-Datensätze bereinigen
    # ----------------------------------------------------------
    print("Schritt 2: Tabellen-Datensätze bereinigen")
    pop_df = clean_population_age(FILE_POPULATION, year=DEFAULT_YEAR)
    hh_df  = clean_households(FILE_HOUSEHOLDS, year=DEFAULT_YEAR)
    sz_df  = clean_population_scenarios(FILE_SCENARIOS)

    # ----------------------------------------------------------
    # 3) Quartier-Feature-Tabelle bauen (Joins auf qnr)
    # ----------------------------------------------------------
    print("Schritt 3: Quartier-Feature-Tabelle bauen")
    quartier_features = (
        quartiere
        .merge(pop_df.drop(columns=["qname"]), on="qnr", how="left")
        .merge(hh_df.drop(columns=["qname"]), on="qnr", how="left")
        .merge(sz_df.drop(columns=["qname"]), on="qnr", how="left")
    )

    # ----------------------------------------------------------
    # 4) Raster erzeugen
    # ----------------------------------------------------------
    print("Schritt 4: Raster erzeugen")
    grid = build_grid(boundary, cell_size_m=CELL_SIZE_M)

    # ----------------------------------------------------------
    # 5) Rasterzellen den Quartieren zuordnen
    # ----------------------------------------------------------
    # Zentroid-Join: für jede Zelle das räumlich nächste Quartier finden
    print("Schritt 5: Rasterzellen → nächstes Quartier")
    centroids = gpd.GeoDataFrame(
        {"cell_id": grid["cell_id"]},
        geometry=gpd.points_from_xy(grid["centroid_x_m"], grid["centroid_y_m"]),
        crs="EPSG:2056",
    )

    joined = gpd.sjoin_nearest(centroids, quartier_features.set_geometry(quartier_features.geometry), how="left")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined.drop_duplicates(subset="cell_id", keep="first")

    feature_cols = [c for c in joined.columns if c not in ("cell_id", "geometry")]
    grid = grid.merge(joined[["cell_id"] + feature_cols], on="cell_id", how="left")

    # ----------------------------------------------------------
    # 6) Punkt-basierte Features (Spitex + ÖV)
    # ----------------------------------------------------------
    def add_point_features(target_grid, target_centroids, points, dist_col, count_col, radius_m):
        # Distanz zum nächsten Punkt berechnen
        nearest = gpd.sjoin_nearest(target_centroids, points[["geometry"]], how="left", distance_col=dist_col)
        nearest = nearest.drop_duplicates(subset="cell_id", keep="first")
        target_grid = target_grid.merge(nearest[["cell_id", dist_col]], on="cell_id", how="left")

        # Anzahl Punkte im Radius (Buffer um Zentroid)
        buffer_gdf = target_centroids.copy()
        buffer_gdf["geometry"] = buffer_gdf.geometry.buffer(radius_m)
        in_buffer = gpd.sjoin(points[["geometry"]], buffer_gdf, how="inner", predicate="within")
        counts = in_buffer.groupby("cell_id").size().rename(count_col).reset_index()
        target_grid = target_grid.merge(counts, on="cell_id", how="left")
        target_grid[count_col] = target_grid[count_col].fillna(0).astype(int)
        return target_grid

    print("Schritt 6a: Distanz + Anzahl Spitex")
    grid = add_point_features(
        grid, centroids, spitex,
        dist_col="dist_nearest_spitex_m",
        count_col=f"n_spitex_{SPITEX_RADIUS_M}m",
        radius_m=SPITEX_RADIUS_M,
    )

    print("Schritt 6b: Distanz + Anzahl ÖV-Haltestellen")
    grid = add_point_features(
        grid, centroids, oev,
        dist_col="dist_nearest_oev_m",
        count_col=f"n_oev_stops_{OEV_RADIUS_M}m",
        radius_m=OEV_RADIUS_M,
    )

    # ----------------------------------------------------------
    # 7) Score-Vorbereitungs-Features (0..1 normiert)
    # ----------------------------------------------------------
    print("Schritt 7: Normierte Score-Features berechnen")

    # Demand-Score: Mittelwert aus Anteil 80+ und Anteil Einpersonenhaushalte
    share_80 = min_max_normalize(grid["share_80plus"])
    share_single = min_max_normalize(grid["share_einpersonen"])
    grid["demand_score"] = ((share_80 + share_single) / 2.0).fillna(0.5)

    # Undersupply-Score: weiter weg von Spitex = höherer Score
    grid["undersupply_score"] = min_max_normalize(grid["dist_nearest_spitex_m"].fillna(0))

    # Accessibility-Score: näher zur ÖV-Haltestelle = höherer Score (invertiert)
    dist_oev_norm = min_max_normalize(grid["dist_nearest_oev_m"].fillna(0))
    grid["accessibility_score"] = (1.0 - dist_oev_norm).fillna(0.5)

    # Growth-Score: Kreis-1-Quartiere haben NaN bei growth_10y → Median als Füllwert
    growth_median = grid["growth_10y"].median()
    growth_filled = grid["growth_10y"].fillna(growth_median)
    grid["growth_score"] = min_max_normalize(growth_filled).fillna(0.5)

    # ----------------------------------------------------------
    # 8) Output als GeoJSON speichern
    # ----------------------------------------------------------
    print("Schritt 8: Output speichern")

    # In WGS84 reprojizieren (Folium erwartet Lat/Lon)
    grid_wgs84 = grid.to_crs("EPSG:4326")

    out = Path(OUT_GEOJSON)
    if out.exists():
        out.unlink()
    grid_wgs84.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"  Gespeichert: {OUT_GEOJSON}")
    print(f"  Rasterzellen: {len(grid):,}  |  Spalten: {len(grid.columns)}")
    print("Nächster Schritt: python scripts/run_clustering.py")


if __name__ == "__main__":
    main()
