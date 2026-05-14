# data_loader.py
# Lädt alle Geo- und Feature-Daten für die Spitex-App.
# @st.cache_data sorgt dafür, dass jede Datei pro Session nur einmal geladen wird –
# ohne Cache würden die GeoPackages bei jedem Slider-Klick neu gelesen.

import geopandas as gpd
import pandas as pd
import streamlit as st


@st.cache_data
def load_spitex():
    # Spitex-Standorte laden und in LV95 reprojizieren
    gdf = gpd.read_file("data/raw/Spitexstandorte-Stadt-ZH.gpkg", layer="stzh.poi_spitex_view")
    gdf = gdf.to_crs("EPSG:2056")
    return gdf[["name", "adresse", "geometry"]]


@st.cache_data
def load_city_boundary():
    # Stadtgrenze Zürich aus den Gemeindegrenzen laden
    gemeinden = gpd.read_file("data/raw/Gemeindegrenzen-ZH.gpkg", layer="up_gemeinden_f")
    zh = gemeinden[gemeinden["gemeindename"] == "Zürich"].copy()
    return zh.to_crs("EPSG:2056")[["gemeindename", "geometry"]]


@st.cache_data
def load_oev_stops():
    # ÖV-Haltestellen laden und in LV95 reprojizieren
    gdf = gpd.read_file("data/raw/Haltestellen_des_offentlichen_Verkehrs_-OGD.gpkg", layer="zvv_haltestellen_p")
    gdf = gdf.to_crs("EPSG:2056")
    return gdf[["chstname", "vtyp", "symb_text", "linien", "geometry"]]


@st.cache_data
def load_grid_features():
    # Geclustertes Grid laden (enthält Features + Cluster-Labels)
    return gpd.read_file("data/processed/grid_features_clustered.geojson")


@st.cache_data
def load_cluster_profiles():
    # Cluster-Profile laden (Mittelwerte pro Cluster)
    return pd.read_csv("data/processed/cluster_profiles.csv")
