# cleaning.py
# Cleaning-Funktionen pro Datensatz.
# Jede Funktion gibt einen kleinen DataFrame aggregiert auf Statistische Quartiere zurück.
# Das gemeinsame Join-Feld heisst immer "qnr".

from pathlib import Path

import pandas as pd

DEFAULT_YEAR = 2025
SZENARIO_YEAR_NOW = 2025
SZENARIO_YEAR_FUTURE = 2035


def clean_population_age(csv_path: Path, year: int = DEFAULT_YEAR) -> pd.DataFrame:
    # Bevölkerungsdaten laden
    df = pd.read_csv(csv_path)

    # Nur das gewünschte Jahr behalten
    df = df[df["StichtagDatJahr"] == year].copy()

    # Nur den letzten Monat verwenden (stabilster Wert)
    latest_month = df["StichtagDatMM"].max()
    df = df[df["StichtagDatMM"] == latest_month]

    # Über Geschlecht und Herkunft summieren – pro Quartier × Altersgruppe
    agg = (
        df.groupby(["QuarCd", "QuarLang", "AlterV20ueber80Kurz_noDM"], as_index=False)
          ["AnzBestWir"].sum()
    )

    # Wide-Format: eine Spalte pro Altersgruppe
    wide = agg.pivot_table(
        index=["QuarCd", "QuarLang"],
        columns="AlterV20ueber80Kurz_noDM",
        values="AnzBestWir",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    # Spaltennamen auf Snake-Case umbenennen
    wide = wide.rename(columns={
        "0-19": "pop_0_19",
        "20-39": "pop_20_39",
        "40-59": "pop_40_59",
        "60-79": "pop_60_79",
        "80 u. älter": "pop_80plus",
    })

    # Gesamtbevölkerung und Anteile berechnen
    wide["pop_total"] = (
        wide["pop_0_19"] + wide["pop_20_39"] + wide["pop_40_59"]
        + wide["pop_60_79"] + wide["pop_80plus"]
    )
    wide["pop_60plus"] = wide["pop_60_79"] + wide["pop_80plus"]

    # Anteile – Schutz vor Division durch 0
    wide["share_60plus"] = (wide["pop_60plus"] / wide["pop_total"]).where(wide["pop_total"] > 0, 0.0)
    wide["share_80plus"] = (wide["pop_80plus"] / wide["pop_total"]).where(wide["pop_total"] > 0, 0.0)

    # Join-Spalten umbenennen
    wide = wide.rename(columns={"QuarCd": "qnr", "QuarLang": "qname"})

    return wide[["qnr", "qname", "pop_total", "pop_60plus", "pop_80plus", "share_60plus", "share_80plus"]].sort_values("qnr").reset_index(drop=True)


def clean_households(csv_path: Path, year: int = DEFAULT_YEAR) -> pd.DataFrame:
    # Haushaltsdaten laden
    df = pd.read_csv(csv_path)

    # Nur das gewünschte Jahr behalten
    df = df[df["StichtagDatJahr"] == year].copy()

    # Einpersonenhaushalte: HHtypSort == 1 (laut Metadaten)
    einpersonen = (
        df[df["HHtypSort"] == 1]
        .groupby(["QuarSort", "QuarLang"], as_index=False)["AnzHH"]
        .sum()
        .rename(columns={"AnzHH": "hh_einpersonen"})
    )

    # Alle Haushalte
    total = (
        df.groupby(["QuarSort", "QuarLang"], as_index=False)["AnzHH"]
        .sum()
        .rename(columns={"AnzHH": "hh_total"})
    )

    # Zusammenführen und Anteil berechnen
    merged = total.merge(einpersonen, on=["QuarSort", "QuarLang"], how="left")
    merged["hh_einpersonen"] = merged["hh_einpersonen"].fillna(0).astype(int)
    merged["share_einpersonen"] = (
        merged["hh_einpersonen"] / merged["hh_total"]
    ).where(merged["hh_total"] > 0, 0.0)

    # Join-Spalten umbenennen
    merged = merged.rename(columns={"QuarSort": "qnr", "QuarLang": "qname"})

    return merged[["qnr", "qname", "hh_total", "hh_einpersonen", "share_einpersonen"]].sort_values("qnr").reset_index(drop=True)


def clean_population_scenarios(
    csv_path: Path,
    year_now: int = SZENARIO_YEAR_NOW,
    year_future: int = SZENARIO_YEAR_FUTURE,
    szenario: str = "mittleres Szenario",
) -> pd.DataFrame:
    # Szenarien-Daten laden
    df = pd.read_csv(csv_path)

    # Nur das mittlere Szenario behalten
    df = df[df["VersionArtLang"] == szenario].copy()

    # Auf die zwei Zieljahre filtern
    df = df[df["StichtagDatJahr"].isin([year_now, year_future])]

    # Über Alter, Geschlecht, Herkunft summieren – pro Quartier × Jahr
    agg = (
        df.groupby(["QuarSort", "QuarLang", "StichtagDatJahr"], as_index=False)
          ["AnzBestWir"].sum()
    )

    # Pivot: eine Spalte pro Jahr
    wide = agg.pivot_table(
        index=["QuarSort", "QuarLang"],
        columns="StichtagDatJahr",
        values="AnzBestWir",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={
        year_now: "pop_szenario_now",
        year_future: "pop_szenario_future",
    })

    # Wachstum in Prozent berechnen
    wide["growth_10y"] = (
        (wide["pop_szenario_future"] - wide["pop_szenario_now"])
        / wide["pop_szenario_now"]
    ).where(wide["pop_szenario_now"] > 0, 0.0)

    # Aggregations-Code 10 = "Unbekannt (Kreis 1)" raus – nicht eindeutig zuordbar
    wide = wide[wide["QuarSort"] != 10]

    # Join-Spalten umbenennen
    wide = wide.rename(columns={"QuarSort": "qnr", "QuarLang": "qname"})

    return wide[["qnr", "qname", "pop_szenario_now", "pop_szenario_future", "growth_10y"]].sort_values("qnr").reset_index(drop=True)
