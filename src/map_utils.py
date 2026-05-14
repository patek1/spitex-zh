# map_utils.py
# Baut die interaktive Folium-Karte für die Spitex-App auf.
#
# Layer auf der Karte:
#   1. Rasterzellen (eingefärbt nach recommendation_score oder cluster)
#   2. Stadtgrenze (Outline)
#   3. Spitex-Standorte (optional)
#   4. ÖV-Haltestellen (optional)
#   5. LayerControl (oben rechts)

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd

from src.clustering import CLUSTER_COLORS, CLUSTER_NAMES

# Karte: Mittelpunkt Zürich + Standard-Zoom
MAP_CENTER = [47.3769, 8.5417]
MAP_ZOOM = 12


def build_map(
    grid_gdf: gpd.GeoDataFrame,
    spitex_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    metric_col: str = "recommendation_score",
    show_spitex: bool = True,
    oev_gdf: gpd.GeoDataFrame = None,
    show_oev: bool = False,
    cluster_filter: int = -1,
    names_df: pd.DataFrame = None,
) -> folium.Map:
    # Alle Layer nach WGS84 reprojizieren (Folium braucht lat/lon)
    grid_wgs = grid_gdf.to_crs("EPSG:4326")
    boundary_wgs = boundary_gdf.to_crs("EPSG:4326")

    # Cluster-Namen aus dem hardcodierten Dict
    cluster_name_map = {cid: info["name"] for cid, info in CLUSTER_NAMES.items()}

    # Basiskarte
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="CartoDB positron")

    # Feste Tooltip-Felder (alle sind in grid_features_clustered.geojson vorhanden)
    tooltip_fields = [
        "cluster_name", "cluster", "cell_id", "qname", "recommendation_score",
        "demand_score", "undersupply_score", "growth_score", "accessibility_score",
        "dist_nearest_spitex_m", "share_80plus",
    ]
    # Nur Felder behalten, die tatsächlich im DataFrame stehen
    tooltip_fields = [f for f in tooltip_fields if f in grid_wgs.columns]

    tooltip_aliases = {
        "cluster_name": "Gebietstyp:",
        "cluster": "Cluster-ID:",
        "cell_id": "Zell-ID:",
        "qname": "Quartier:",
        "recommendation_score": "Empfehlungsscore:",
        "demand_score": "Pflegebedarf:",
        "undersupply_score": "Unterversorgung:",
        "growth_score": "Wachstum:",
        "accessibility_score": "Erreichbarkeit:",
        "dist_nearest_spitex_m": "Distanz Spitex (m):",
        "share_80plus": "Anteil 80+:",
    }
    aliases = [tooltip_aliases.get(f, f + ":") for f in tooltip_fields]

    # Spalten, die für GeoJSON benötigt werden
    cols_for_json = list(set(tooltip_fields + [metric_col, "geometry"]))
    cols_for_json = [c for c in cols_for_json if c in grid_wgs.columns]

    # --- Fall 1: kategoriale Cluster-Farben ---
    if metric_col == "cluster":
        def style_function_cluster(feature):
            cluster_id = feature["properties"].get("cluster")
            if cluster_id is None:
                return {"fillColor": "#aaaaaa", "fillOpacity": 0.4, "color": "none", "weight": 0}
            cluster_id = int(cluster_id)
            # Nicht ausgewählte Cluster grau und transparent darstellen
            if cluster_filter >= 0 and cluster_id != cluster_filter:
                return {"fillColor": "#cccccc", "fillOpacity": 0.2, "color": "none", "weight": 0}
            color = CLUSTER_COLORS.get(cluster_id, "#aaaaaa")
            return {"fillColor": color, "fillOpacity": 0.7, "color": "none", "weight": 0}

        raster_layer = folium.GeoJson(
            grid_wgs[cols_for_json].to_json(),
            name="Raster (Gebietstypen)",
            style_function=style_function_cluster,
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases, localize=True),
        )

        # Manuelle HTML-Legende (Folium hat keine kategoriale Farbskala)
        legend_html = _build_cluster_legend(grid_wgs, cluster_filter, cluster_name_map)
        m.get_root().html.add_child(folium.Element(legend_html))

    # --- Fall 2: kontinuierliche Farbskala (Empfehlungsscore) ---
    else:
        colormap = cm.LinearColormap(
            colors=["#2ecc71", "#f1c40f", "#e74c3c"],
            vmin=0.0,
            vmax=1.0,
            caption="Empfehlungsscore (0 = niedrig, 1 = hoch)",
        )

        def style_function_numeric(feature):
            value = feature["properties"].get(metric_col, 0.5)
            if value is None:
                value = 0.5
            value = max(0.0, min(1.0, float(value)))
            return {"fillColor": colormap(value), "fillOpacity": 0.65, "color": "none", "weight": 0}

        raster_layer = folium.GeoJson(
            grid_wgs[cols_for_json].to_json(),
            name="Raster (Empfehlungsscore)",
            style_function=style_function_numeric,
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases, localize=True),
        )
        colormap.add_to(m)

    raster_layer.add_to(m)

    # --- Stadtgrenze (nur Outline) ---
    if boundary_wgs is not None and len(boundary_wgs) > 0:
        folium.GeoJson(
            boundary_wgs.to_json(),
            name="Stadtgrenze Zürich",
            style_function=lambda _: {"color": "#2c3e50", "weight": 2.5, "fillOpacity": 0},
        ).add_to(m)

    # --- Spitex-Standorte als Marker ---
    if show_spitex and spitex_gdf is not None and len(spitex_gdf) > 0:
        spitex_wgs = spitex_gdf.to_crs("EPSG:4326")
        spitex_layer = folium.FeatureGroup(name="Spitex-Standorte", show=True)

        for _, row in spitex_wgs.iterrows():
            if row.geometry is None:
                continue
            lat, lon = row.geometry.y, row.geometry.x
            name = row.get("name", "Spitex-Standort")
            adresse = row.get("adresse", "")
            popup_html = f"<b>{name}</b>"
            if adresse:
                popup_html += f"<br>{adresse}"
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color="#2980b9",
                fill=True,
                fill_color="#3498db",
                fill_opacity=0.9,
                tooltip=name,
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(spitex_layer)

        spitex_layer.add_to(m)

    # --- ÖV-Haltestellen als kleine Marker ---
    if show_oev and oev_gdf is not None and len(oev_gdf) > 0:
        oev_wgs = oev_gdf.to_crs("EPSG:4326")
        oev_layer = folium.FeatureGroup(name="ÖV-Haltestellen", show=True)

        type_colors = {
            "Bahn": "#8e44ad",
            "Tram": "#27ae60",
            "Bus Tram": "#16a085",
            "Bus": "#7f8c8d",
        }
        for _, row in oev_wgs.iterrows():
            if row.geometry is None:
                continue
            lat, lon = row.geometry.y, row.geometry.x
            sym = row.get("symb_text", "") or ""
            color = type_colors.get(sym, "#95a5a6")
            name = row.get("chstname", "ÖV-Haltestelle")
            linien = row.get("linien", "")
            popup = f"<b>{name}</b>"
            if sym:
                popup += f"<br>Typ: {sym}"
            if linien:
                popup += f"<br>Linien: {linien}"
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1,
                tooltip=name,
                popup=folium.Popup(popup, max_width=240),
            ).add_to(oev_layer)

        oev_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _build_cluster_legend(
    grid_wgs: gpd.GeoDataFrame,
    cluster_filter: int,
    cluster_name_map: dict,
) -> str:
    # HTML-Legende mit Cluster-Namen und Farben (Folium hat keine kategoriale Skala)
    cluster_ids = sorted(grid_wgs["cluster"].dropna().unique().astype(int).tolist())

    items_html = ""
    for cid in cluster_ids:
        color = CLUSTER_COLORS.get(cid, "#aaaaaa")
        is_active = cluster_filter < 0 or cid == cluster_filter
        weight = "bold" if is_active else "normal"
        opacity = "1.0" if is_active else "0.4"
        label = cluster_name_map.get(cid, f"Cluster {cid}")
        items_html += (
            f'<div style="display:flex; align-items:flex-start; margin:4px 0; opacity:{opacity};">'
            f'<div style="background:{color}; min-width:14px; height:14px; '
            f'border-radius:3px; margin-right:8px; margin-top:2px; flex-shrink:0;"></div>'
            f'<span style="font-weight:{weight}; font-size:11px; line-height:1.4;">{label}</span>'
            f"</div>"
        )

    return f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background-color: white; padding: 10px 14px; border-radius: 6px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.25); font-family: sans-serif; max-width: 220px;
    ">
        <div style="font-weight:bold; margin-bottom:6px; font-size:12px;">Gebietstypen</div>
        {items_html}
    </div>
    """
