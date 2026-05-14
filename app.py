# app.py
# Streamlit-Hauptdatei der Spitex Standortplanungs-App.
# Starten mit:  streamlit run app.py
#
# Ablauf:
#   1. Geodaten + Feature-Daten laden (gecacht)
#   2. Sidebar: Standortstrategie + Gewichtungen, Cluster-Filter, Kartenlayer
#   3. Empfehlungsscore live berechnen
#   4. Vier Tabs: Karte | Gebietstypen | Top Standorte | Methodik

import streamlit as st

from src.clustering import CLUSTER_NAMES, attach_cluster_name
from src.data_loader import (
    load_city_boundary,
    load_cluster_profiles,
    load_grid_features,
    load_oev_stops,
    load_spitex,
)
from src.scoring import SCORE_COMPONENT_LABELS, STRATEGIES, compute_recommendation_score
from src.ui_tabs import (
    render_cluster_tab,
    render_map_tab,
    render_methodik_tab,
    render_top_tab,
)

# ----------------------------------------------------------------
# Seiteneinstellungen
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Spitex Standortplanung Zürich",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Spitex Standortplanung Zürich")
st.markdown(
    """
    Das **K-Means Clustering** erkennt datengetriebene Gebietstypen in Zürich.
    Der **Empfehlungsscore** nutzt diese Gebietstypen und die zugrundeliegenden Features,
    um potenziell relevante Standorte je nach Strategie zu priorisieren.
    Die App ist eine **datenbasierte Entscheidungsunterstützung** für die Standortwahl von neuen Spitex-Standorten.
    """
)

# ----------------------------------------------------------------
# Geodaten und Feature-Daten laden (gecacht – nur beim ersten Aufruf)
# ----------------------------------------------------------------
with st.spinner("Daten werden geladen ..."):
    spitex_gdf = load_spitex()
    boundary_gdf = load_city_boundary()
    oev_gdf = load_oev_stops()
    grid_gdf = load_grid_features()
    cluster_profiles = load_cluster_profiles()

# Cluster-Namen dem Grid hinzufügen (für Tooltips + Top-Tabelle)
grid_gdf = attach_cluster_name(grid_gdf)

# Cluster-IDs für den Sidebar-Filter sammeln
cluster_ids = sorted(grid_gdf["cluster"].dropna().unique().astype(int).tolist())

# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Einstellungen")

    # ----------------------------------------------------------
    # Sektion 1: Standortstrategie + Empfehlungsscore-Gewichte
    # ----------------------------------------------------------
    st.subheader("🎯 Standortstrategie")
    st.caption(
        "Wähle eine Strategie oder passe die Gewichte manuell an. "
        "Sie beeinflussen den Empfehlungsscore auf der Karte und in der Top-Tabelle."
    )

    # Strategie auswählen – setzt die Slider-Startwerte
    strategy_names = list(STRATEGIES.keys())
    selected_strategy = st.selectbox(
        "Strategie",
        options=strategy_names,
        index=0,
        help="Preset-Gewichtungen für die vier Score-Komponenten.",
    )

    # Startwerte aus der gewählten Strategie laden
    strategy_weights = STRATEGIES[selected_strategy]

    st.markdown("**Gewichtung anpassen** (optional):")

    # Slider für jede der 4 Score-Komponenten (0–100, Step 5)
    w_demand = st.slider(
        SCORE_COMPONENT_LABELS["demand"],
        min_value=0, max_value=100, step=5,
        value=int(strategy_weights["demand"]),
        help="Anteil 80+ und Einpersonenhaushalte",
    )
    w_undersupply = st.slider(
        SCORE_COMPONENT_LABELS["undersupply"],
        min_value=0, max_value=100, step=5,
        value=int(strategy_weights["undersupply"]),
        help="Distanz zur nächsten Spitex (grösser = unterversorgter)",
    )
    w_growth = st.slider(
        SCORE_COMPONENT_LABELS["growth"],
        min_value=0, max_value=100, step=5,
        value=int(strategy_weights["growth"]),
        help="Bevölkerungswachstum 2025–2035",
    )
    w_accessibility = st.slider(
        SCORE_COMPONENT_LABELS["accessibility"],
        min_value=0, max_value=100, step=5,
        value=int(strategy_weights["accessibility"]),
        help="Nähe zur ÖV-Haltestelle (näher = besser erreichbar)",
    )

    # Gewichtssumme anzeigen
    total_w = w_demand + w_undersupply + w_growth + w_accessibility
    if total_w == 0:
        st.warning("Alle Gewichte sind 0 – bitte mindestens einen Wert > 0 setzen.")
    else:
        st.caption(f"Summe: {total_w} (wird intern auf 100 % normiert)")

    st.divider()

    # ----------------------------------------------------------
    # Sektion 2: Cluster-Filter
    # ----------------------------------------------------------
    st.subheader("🗂️ Gebietstyp-Filter")

    # Optionen: "Alle" + ein Eintrag pro Cluster mit Namen
    cluster_options: dict[str, int] = {"Alle Gebietstypen anzeigen": -1}
    for cid in cluster_ids:
        label = CLUSTER_NAMES.get(cid, {}).get("name", f"Cluster {cid}")
        cluster_options[label] = cid

    selected_cluster_label = st.selectbox(
        "Gebietstyp auswählen",
        options=list(cluster_options.keys()),
        help="Filtert Karte und Top-Standorte auf einen Gebietstyp.",
    )
    selected_cluster = cluster_options[selected_cluster_label]

    st.divider()

    # ----------------------------------------------------------
    # Sektion 3: Kartenanzeige
    # ----------------------------------------------------------
    st.subheader("🗺️ Kartenanzeige")

    METRIC_OPTIONS = {
        "Empfehlungsscore": "recommendation_score",
        "Gebietstypen (Cluster)": "cluster",
    }
    selected_metric_label = st.selectbox(
        "Kartenart",
        options=list(METRIC_OPTIONS.keys()),
        index=0,
    )
    selected_metric_col = METRIC_OPTIONS[selected_metric_label]

    show_spitex = st.checkbox("Spitex-Standorte anzeigen", value=True)
    show_oev = st.checkbox("ÖV-Haltestellen anzeigen", value=False)

    st.divider()
    st.subheader("ℹ️ Datenquellen")
    st.markdown(
        """
        **Geodaten (GeoPackage)**
        - [Spitex-Standorte](https://www.stadt-zuerich.ch/geodaten/download/Spitex) – Stadt Zürich
        - [Gemeindegrenzen (OGD)](https://www.stadt-zuerich.ch/geodaten/download/95) – Stadt Zürich
        - [Statistische Quartiere](https://www.stadt-zuerich.ch/geodaten/download/Statistische_Quartiere) – Stadt Zürich
        - [Haltestellen ÖV (OGD)](https://data.stadt-zuerich.ch/dataset/ktzh_haltestellen_des_oeffentlichen_verkehrs___ogd_) – Kanton ZH

        **Sachdaten (Open Data Zürich)**
        - [Bevölkerung nach Altersgruppe & Quartier](https://data.stadt-zuerich.ch/dataset/bev_monat_bestand_quartier_geschl_ag_herkunft_od3250)
        - [Privathaushalte nach Haushaltsform & Quartier](https://data.stadt-zuerich.ch/dataset/bev_hh_haushaltsform_quartier_seit2013_od3805)
        - [Bevölkerungsszenarien 1993–2050](https://data.stadt-zuerich.ch/dataset/bev_szenarien_od3440)
        """
    )

# ----------------------------------------------------------------
# Empfehlungsscore live berechnen
# ----------------------------------------------------------------
# Jedes Mal neu berechnen, wenn der User einen Slider bewegt
user_weights = {
    "demand": w_demand,
    "undersupply": w_undersupply,
    "growth": w_growth,
    "accessibility": w_accessibility,
}
grid_gdf["recommendation_score"] = compute_recommendation_score(grid_gdf, user_weights)

# ----------------------------------------------------------------
# Grid nach gewähltem Cluster filtern
# ----------------------------------------------------------------
if selected_cluster >= 0:
    grid_filtered = grid_gdf[grid_gdf["cluster"] == selected_cluster].copy()
else:
    grid_filtered = grid_gdf.copy()

# ----------------------------------------------------------------
# KPI-Zeile
# ----------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rasterzellen total", f"{len(grid_gdf):,}")
col2.metric("Spitex-Standorte", f"{len(spitex_gdf)}")
col3.metric(
    "Zellen im Filter",
    f"{len(grid_filtered):,}",
    delta=f"{'Gebietstyp: ' + selected_cluster_label if selected_cluster >= 0 else 'alle'}",
    delta_color="off",
)
col4.metric("Strategie", selected_strategy)

# ----------------------------------------------------------------
# Vier Tabs
# ----------------------------------------------------------------
tab_karte, tab_cluster, tab_top, tab_methodik = st.tabs(
    ["🗺️ Karte", "🎯 Gebietstypen", "🏆 Top Standorte", "📖 Methodik"]
)

with tab_karte:
    render_map_tab(
        grid_gdf=grid_gdf,
        spitex_gdf=spitex_gdf,
        boundary_gdf=boundary_gdf,
        oev_gdf=oev_gdf,
        metric_col=selected_metric_col,
        metric_label=selected_metric_label,
        show_spitex=show_spitex,
        show_oev=show_oev,
        cluster_filter=selected_cluster,
    )

with tab_cluster:
    render_cluster_tab(
        grid_gdf=grid_gdf,
        cluster_profiles=cluster_profiles,
    )

with tab_top:
    render_top_tab(
        grid_filtered=grid_filtered,
        selected_cluster=selected_cluster,
    )

with tab_methodik:
    render_methodik_tab(grid_gdf=grid_gdf)
