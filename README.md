# 🏥 Spitex Standortplanung Zürich

Interaktive Streamlit-App zur datenbasierten Identifikation potenzieller neuer **Spitex-Standorte in der Stadt Zürich**.
Entwickelt als Capstone-Projekt im Rahmen des Kurses *Methoden: Data Science und AI for Business* an der Universität St. Gallen (HSG).

> Repo: [github.com/patek1/spitex-zh](https://github.com/patek1/spitex-zh)

---

## Ziel

Die ambulante Pflege (**Spitex**) muss in Zürich mit einer alternden Bevölkerung und steigender Nachfrage umgehen.
Standortentscheidungen werden oft ohne räumliche Datenanalyse getroffen.

**Frage:** *Wo in Zürich gibt es Quartiere mit hohem Pflegebedarf, die weit von einer Spitex-Filiale entfernt sind
und sich daher als Standort für eine neue Filiale anbieten?*

Diese App bietet Entscheidungsträger:innen einen datenbasierten ersten Blick auf das Stadtgebiet –
als Diskussionswerkzeug, nicht als finale Standortempfehlung.

---

## Was die App kann

- **K-Means Clustering** als Haupt-ML-Methode: 2'466 Rasterzellen (200×200 m) werden anhand demografischer
  und räumlicher Features in 4 Gebietstypen gruppiert
- **Interaktiver Empfehlungsscore** mit frei wählbaren Gewichten (Pflegebedarf, Unterversorgung, Wachstum, Erreichbarkeit)
- **Interaktive Karte** mit Cluster-Färbung oder Empfehlungsscore-Layer
- **Gebietstypen-Tab** mit Beschreibungen und Mittelwerten pro Cluster
- **Top-10-Tabelle** der Rasterzellen mit höchstem Empfehlungsscore

---

## Projektstruktur

```
spitex-app/
├── app.py                       # Streamlit-App (Einstiegspunkt)
├── requirements.txt
├── data/
│   ├── raw/                     # Rohdaten (GeoPackages + CSVs)
│   └── processed/               # Aufbereitete Daten (werden von Skripten erzeugt)
├── scripts/
│   ├── build_features.py        # Feature-Pipeline: Rohdaten → grid_features.geojson
│   └── run_clustering.py        # Clustering-Pipeline: Features → Cluster + Profile
├── src/
│   ├── cleaning.py              # Bereinigung der CSV-Datensätze
│   ├── clustering.py            # K-Means Hilfsfunktionen + Cluster-Namen
│   ├── data_loader.py           # Geodaten + Feature-Datei lesen
│   ├── map_utils.py             # Folium-Karte aufbauen
│   ├── scoring.py               # Empfehlungsscore (Strategien + Berechnung)
│   └── ui_tabs.py               # Streamlit-Tab-Funktionen
└── notebooks/
    ├── 01_clustering_exploration.ipynb   # Clustering: Silhouette, Profile, PCA
    └── 02_supervised_ml.ipynb            # Supervised ML: Klassifikation + Regression
```

---

## Daten

Alle Daten von [Open Data Zürich](https://data.stadt-zuerich.ch) – Lizenz CC Zero.

### Geodaten (`.gpkg`)

| Datei | Quelle |
|---|---|
| `Spitexstandorte-Stadt-ZH.gpkg` | [Spitex-Standorte](https://www.stadt-zuerich.ch/geodaten/download/Spitex) |
| `Gemeindegrenzen-ZH.gpkg` | [Gemeindegrenzen](https://www.stadt-zuerich.ch/geodaten/download/95) |
| `Statistische-Quartiere-ZH.gpkg` | [Statistische Quartiere](https://www.stadt-zuerich.ch/geodaten/download/Statistische_Quartiere) |
| `Haltestellen_des_offentlichen_Verkehrs_-OGD.gpkg` | [Haltestellen ÖV](https://data.stadt-zuerich.ch/dataset/ktzh_haltestellen_des_oeffentlichen_verkehrs___ogd_) |

### Sachdaten (`.csv`)

| Datei | Quelle |
|---|---|
| `Bevölkerung-Altergruppe-Quartier-ZH.csv` | [Bevölkerung nach Altersgruppe & Quartier](https://data.stadt-zuerich.ch/dataset/bev_monat_bestand_quartier_geschl_ag_herkunft_od3250) |
| `Privathaushalte-Haushaltsform-Stadtquartier-ZH.csv` | [Privathaushalte nach Haushaltsform](https://data.stadt-zuerich.ch/dataset/bev_hh_haushaltsform_quartier_seit2013_od3805) |
| `Bevölkerungsszenarien.csv` | [Bevölkerungsszenarien 1993–2050](https://data.stadt-zuerich.ch/dataset/bev_szenarien_od3440) |

---

## ML-Methode: K-Means Clustering

Da es kein bekanntes Label für "optimaler Spitex-Standort" gibt, verwenden wir **K-Means Clustering** (unsupervised Learning).

### Features (9 Variablen)

| Feature | Bedeutung |
|---|---|
| `share_60plus` | Anteil 60+: Nachfrage-Indikator |
| `share_80plus` | Anteil 80+: starker Pflegebedarf |
| `share_einpersonen` | Anteil Einpersonenhaushalte: Pflegerisiko |
| `growth_10y` | Bevölkerungswachstum 2025–2035 |
| `pop_total` | Bevölkerung total |
| `dist_nearest_spitex_m` | Distanz zur nächsten Spitex |
| `n_spitex_1000m` | Anzahl Spitex im 1-km-Umkreis |
| `dist_nearest_oev_m` | Distanz zur nächsten ÖV-Haltestelle |
| `n_oev_stops_500m` | Anzahl ÖV-Stops im 500-m-Umkreis |

### Preprocessing

1. Fehlende Werte → Median-Imputation
2. Skalierung → StandardScaler (Mittelwert = 0, Std = 1)

### k-Wahl

k = 3, 4, 5, 6 getestet. Bestes Ergebnis nach Silhouette-Score: k=4.

---

## Empfehlungsscore

Der Score kombiniert vier normierte Komponenten (0..1) zu einer einzigen Kennzahl pro Rasterzelle:

```
score = w_Pflegebedarf × demand_score
      + w_Unterversorgung × undersupply_score
      + w_Wachstum × growth_score
      + w_Erreichbarkeit × accessibility_score
```

Die Gewichte (w) lassen sich in der Sidebar interaktiv anpassen.

---

## Installation & Ausführung

### 1. Repo klonen

```bash
git clone https://github.com/patek1/spitex-zh.git
cd spitex-zh
```

### 2. Virtuelle Umgebung erstellen und aktivieren

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### 3. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 4. Rohdaten herunterladen

Dateien aus den Quellen oben herunterladen und in `data/raw/` ablegen.

### 5. Pipelines ausführen

```bash
python scripts/build_features.py
python scripts/run_clustering.py
```

### 6. App starten

```bash
streamlit run app.py
```

Die App öffnet sich im Browser unter `http://localhost:8501`.

### 7. Notebooks öffnen (optional)

```bash
jupyter notebook notebooks/01_clustering_exploration.ipynb
jupyter notebook notebooks/02_supervised_ml.ipynb
```

---

## Tech-Stack

- **Streamlit** – Web-App-Framework
- **scikit-learn** – K-Means, StandardScaler, Silhouette Score
- **Folium** + **streamlit-folium** – interaktive Karten
- **GeoPandas / Shapely** – Geodatenverarbeitung (Swiss LV95 EPSG:2056)
- **Pandas / NumPy** – Datenverarbeitung
- **Matplotlib / Seaborn** – Visualisierungen

---

*Universität St. Gallen (HSG) | Methoden: Data Science und AI for Business | Capstone-Projekt*
