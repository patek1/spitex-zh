# ui_tabs.py
# Render-Funktionen für die vier Streamlit-Tabs der Spitex-App.

import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.clustering import CLUSTER_COLORS, CLUSTER_NAMES, CLUSTERING_FEATURES
from src.map_utils import build_map


# -------------------------------------------------------------------
# Tab 1: Karte
# -------------------------------------------------------------------
def render_map_tab(
    grid_gdf: gpd.GeoDataFrame,
    spitex_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    oev_gdf: gpd.GeoDataFrame,
    metric_col: str,
    metric_label: str,
    show_spitex: bool,
    show_oev: bool,
    cluster_filter: int,
) -> None:
    # Caption unter dem Karten-Tab
    caption_parts = [f"Kartenart: **{metric_label}**"]
    if cluster_filter >= 0:
        caption_parts.append(f"Filter: Cluster {cluster_filter}")
    if metric_col == "cluster":
        caption_parts.append("Farben = Gebietstypen (K-Means Cluster)")
    elif metric_col == "recommendation_score":
        caption_parts.append("Grün = niedrig · Gelb = mittel · Rot = hoch")
    st.caption(" | ".join(caption_parts) + ".")

    # Karte bauen und anzeigen
    karte = build_map(
        grid_gdf,
        spitex_gdf,
        boundary_gdf,
        metric_col=metric_col,
        show_spitex=show_spitex,
        oev_gdf=oev_gdf,
        show_oev=show_oev,
        cluster_filter=cluster_filter,
    )
    st_folium(karte, use_container_width=True, height=600, returned_objects=[])

    # Kurze Erklärung unter der Karte
    if metric_col == "recommendation_score":
        st.info(
            "**Empfehlungsscore**: Jede Rasterzelle erhält einen Wert zwischen 0 und 1, "
            "berechnet aus Pflegebedarf, Unterversorgung, Wachstum und Erreichbarkeit. "
            "Die Gewichtung kannst du in der Seitenleiste anpassen."
        )
    elif metric_col == "cluster":
        st.info(
            "**Gebietstypen**: Die Farben zeigen das K-Means Clustering – jede Farbe steht "
            "für einen datengetriebenen Gebietstyp. Die Bedeutung der Typen findest du im Tab **Gebietstypen**."
        )


# -------------------------------------------------------------------
# Tab 2: Cluster / Gebietstypen
# -------------------------------------------------------------------
def render_cluster_tab(
    grid_gdf: gpd.GeoDataFrame,
    cluster_profiles: pd.DataFrame,
) -> None:
    st.subheader("Gebietstypen: K-Means Clustering")

    # Erklärung K-Means in einfachen Worten
    n_zellen_total = len(grid_gdf)
    n_cluster = len(CLUSTER_NAMES)
    st.markdown(
        f"""
        **Was ist K-Means und was macht es hier?**

        Stell dir vor, du hast {n_zellen_total:,} Rasterzellen in Zürich – jede mit Informationen wie
        Anteil älterer Bevölkerung, Distanz zu einer Spitex oder Bevölkerungswachstum.
        K-Means sucht automatisch **{n_cluster} Gruppen** (Cluster), so dass Zellen innerhalb
        einer Gruppe möglichst ähnlich sind und Zellen aus verschiedenen Gruppen
        möglichst unterschiedlich. Das Modell lernt diese Gruppen aus den Daten –
        ohne dass wir ihm sagen, welche Gebiete "gut" oder "schlecht" sind.

        > **Wichtig:** Die Cluster zeigen *Gebietstypen*, keine Rangliste.
        > Den Empfehlungsscore (Tab *Karte* / *Top Standorte*) nutzen wir als separate
        > Entscheidungshilfe.
        """
    )

    # Silhouette-Score für k=4 anzeigen
    scores_df = pd.read_csv("data/processed/clustering_scores.csv")
    sil_k4 = scores_df[scores_df["k"] == 4]["silhouette_score"].values[0]
    st.caption(
        f"Silhouette-Score für k=4: **{sil_k4:.4f}** "
        "(Wert zwischen -1 und 1, höher = besser getrennte Cluster)"
    )

    st.divider()

    # Gebietstypen als farbige Karten
    st.markdown("#### Gebietstypen im Überblick")
    for cid, info in CLUSTER_NAMES.items():
        color = CLUSTER_COLORS.get(cid, "#aaaaaa")
        n_zellen = len(grid_gdf[grid_gdf["cluster"] == cid])
        st.markdown(
            f'<div style="border-left: 5px solid {color}; padding: 8px 14px; '
            f'margin: 8px 0; background: #f9f9f9; border-radius: 4px;">'
            f'<strong>{info["name"]}</strong> '
            f'<span style="font-size:12px; color:#666;">({n_zellen} Rasterzellen)</span><br>'
            f'<span style="font-size:13px;">{info["description"]}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Cluster-Profil-Tabelle (Mittelwerte pro Cluster)
    st.markdown("#### Merkmals-Mittelwerte pro Gebietstyp")
    st.markdown(
        "Diese Tabelle zeigt den durchschnittlichen Wert jedes Features pro Cluster. "
        "Daran sieht man, welche Merkmale einen Gebietstyp auszeichnen."
    )

    # Leserliche Spaltennamen
    rename_profile = {
        "n_zellen": "Anzahl Zellen",
        "share_60plus": "Anteil 60+",
        "share_80plus": "Anteil 80+",
        "share_einpersonen": "Anteil Einpers.-HH",
        "growth_10y": "Wachstum 10J",
        "pop_total": "Bevölkerung",
        "dist_nearest_spitex_m": "Ø Distanz Spitex (m)",
        "n_spitex_1000m": "Ø Anzahl Spitex 1km",
        "dist_nearest_oev_m": "Ø Distanz ÖV (m)",
        "n_oev_stops_500m": "Ø ÖV-Stops 500m",
    }

    display_profiles = cluster_profiles.copy()

    # Cluster-ID durch Namen ersetzen
    name_map = {cid: info["name"] for cid, info in CLUSTER_NAMES.items()}
    display_profiles["cluster"] = display_profiles["cluster"].apply(
        lambda x: name_map.get(int(x), f"Cluster {int(x)}")
    )
    display_profiles = display_profiles.rename(columns={"cluster": "Gebietstyp"})
    display_profiles = display_profiles.rename(columns={k: v for k, v in rename_profile.items() if k in display_profiles.columns})
    display_profiles = display_profiles.set_index("Gebietstyp")

    st.dataframe(display_profiles.style.format(precision=3), use_container_width=True)


# -------------------------------------------------------------------
# Tab 3: Top Standorte
# -------------------------------------------------------------------
def render_top_tab(
    grid_filtered: gpd.GeoDataFrame,
    selected_cluster: int,
) -> None:
    # Überschrift mit Cluster-Namen oder "alle"
    if selected_cluster >= 0:
        cluster_label = CLUSTER_NAMES.get(selected_cluster, {}).get("name", f"Cluster {selected_cluster}")
    else:
        cluster_label = "alle Gebietstypen"
    st.subheader(f"Top 10 Standorte – {cluster_label}")

    st.markdown(
        "Die Tabelle zeigt die **10 Rasterzellen mit dem höchsten Empfehlungsscore** "
        "im gewählten Filter. Der Score basiert auf deinen Gewichtungen in der Seitenleiste. "
        "Diese Zellen sind **Hinweise für vertiefte Standortprüfungen** – "
        "keine finale Standortentscheidung."
    )

    # Anzeigespalten (alle sind im geclusterten Grid vorhanden)
    anzeige_cols = [
        "cell_id", "qname", "cluster_name", "recommendation_score",
        "demand_score", "undersupply_score", "growth_score", "accessibility_score",
        "dist_nearest_spitex_m", "share_80plus",
    ]
    # Nur Spalten behalten, die wirklich im DataFrame sind
    anzeige_cols = [c for c in anzeige_cols if c in grid_filtered.columns]

    # Top 10 nach Empfehlungsscore
    top10 = (
        grid_filtered[anzeige_cols]
        .sort_values("recommendation_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Leserliche Spaltennamen
    top10 = top10.rename(columns={
        "cell_id": "Zell-ID",
        "qname": "Quartier",
        "cluster_name": "Gebietstyp",
        "recommendation_score": "Empfehlungsscore",
        "demand_score": "Pflegebedarf",
        "undersupply_score": "Unterversorgung",
        "growth_score": "Wachstum",
        "accessibility_score": "Erreichbarkeit",
        "dist_nearest_spitex_m": "Distanz Spitex (m)",
        "share_80plus": "Anteil 80+",
    })
    st.dataframe(top10.style.format(precision=3), use_container_width=True, hide_index=True)

    st.caption(
        "Hinweis: Diese Zellen sind potenziell relevante Gebiete für eine Standortprüfung. "
        "Eine datenbasierte Empfehlung ersetzt keine Vor-Ort-Analyse."
    )


# -------------------------------------------------------------------
# Tab 4: Methodik
# -------------------------------------------------------------------
def render_methodik_tab(grid_gdf: gpd.GeoDataFrame) -> None:
    st.subheader("Methodik: Datenbasierte Standortplanung")

    n_features = len([f for f in CLUSTERING_FEATURES if f in grid_gdf.columns])

    st.markdown(
        f"""
        ### Überblick: Zwei-Schichten-Ansatz

        Diese App nutzt zwei klar getrennte Methoden:

        | Schicht | Methode | Zweck |
        |---|---|---|
        | ML-Part | K-Means Clustering | Erkennt datengetriebene Gebietstypen |
        | Decision Layer | Empfehlungsscore | Priorisiert Gebiete nach Strategie |

        > Das Clustering "sieht" keine Zielgrösse – es gruppiert Gebiete nur nach Ähnlichkeit.
        > Der Empfehlungsscore ist eine transparente Formel, die du selbst steuern kannst.

        ---

        ### Teil 1: K-Means Clustering (ML)

        **Warum Clustering?**
        Es gibt kein bekanntes Label dafür, welche Gebiete sich ideal für neue Spitex-Standorte
        eignen. Deshalb verwenden wir
        **unsupervised Clustering** als ML-Methode.

        Das K-Means Clustering gruppiert die **{len(grid_gdf):,} Rasterzellen**
        (200×200 m, Stadtgebiet Zürich) so, dass ähnliche Zellen im gleichen Cluster landen.
        "Ähnlichkeit" wird über standardisierte Merkmale gemessen (Euklidische Distanz).

        **Features fürs Clustering ({n_features} Variablen):**

        | Feature | Bedeutung |
        |---|---|
        | Anteil 60+ / 80+ | Typische Spitex-Nachfrage-Indikatoren |
        | Anteil Einpersonenhaushalte | Alleinstehende = höheres Pflegerisiko |
        | Bevölkerungswachstum 2025–2035 | Künftige Nachfrageentwicklung |
        | Bevölkerung total | Grösse des Quartiers |
        | Distanz zur nächsten Spitex | Versorgungslücke |
        | Anzahl Spitex im 1-km-Umkreis | Lokale Versorgungsdichte |
        | Distanz zur nächsten ÖV-Haltestelle | Erreichbarkeit |
        | Anzahl ÖV-Stops im 500-m-Umkreis | ÖV-Dichte |

        **Preprocessing:** Fehlende Werte → Median-Impute. Features → StandardScaler (Z-Score).

        **Wahl von k:** k=3,4,5,6 wurden mit Silhouette-Score verglichen. k=4 zeigt den
        besten Score und liefert interpretierbare Gruppen für Zürich.

        ---

        ### Teil 2: Empfehlungsscore (Decision Layer)

        Der Empfehlungsscore kombiniert die vier Score-Komponenten zu einer einzigen Kennzahl
        pro Rasterzelle. Die Gewichte bestimmst du selbst über die Strategie in der Seitenleiste:

        ```
        recommendation_score =
            w_Pflegebedarf    × demand_score
          + w_Unterversorgung × undersupply_score
          + w_Wachstum        × growth_score
          + w_Erreichbarkeit  × accessibility_score
        ```

        Alle Score-Komponenten sind auf 0..1 normiert. Die Gewichte (w) wählst du in der
        Seitenleiste – entweder über eine Strategie oder manuell mit Slidern.

        **Score-Komponenten:**

        | Komponente | Grundlage | Formel |
        |---|---|---|
        | Pflegebedarf | share_80plus + share_einpersonen | Mittelwert, min-max normiert |
        | Unterversorgung | dist_nearest_spitex_m | min-max normiert (grösser = unterversorgter) |
        | Wachstum | growth_10y | min-max normiert |
        | Erreichbarkeit | dist_nearest_oev_m | 1 − min-max normiert (näher = besser erreichbar) |

        ---

        ### Wichtige Einschränkungen

        - **Keine perfekte Vorhersage**: Diese App identifiziert *potenziell relevante Gebiete*,
          keine definitiven Standorte.
        - **Granularität**: Features wie Bevölkerungsanteile sind auf Quartiersebene – alle
          Rasterzellen im gleichen Quartier haben denselben Wert.
        - **Fehlende Daten**: Bevölkerungsszenarien fehlen für 4 Quartiere in Kreis 1
          → growth_score mit Median aufgefüllt.
        - **Entscheidungsunterstützung**: Die Empfehlungen der App sind Hinweise für vertiefte
          Standortprüfungen – sie ersetzen keine Analyse vor Ort.

        ---

        ### Datenquellen

        Alle Daten: [Open Data Zürich](https://data.stadt-zuerich.ch) – Lizenz CC0.
        Raster: 200×200 m, Swiss LV95 (EPSG:2056). Karte: WGS84 (EPSG:4326).

        *Universität St. Gallen (HSG) | Methoden: Data Science und AI for Business | Capstone-Projekt*
        """
    )
