"""
generate_api_data.py
====================
Converts CSV data files into optimized JSON for the Nuxt dashboard frontend.

Outputs:
  dashboard/public/data/market.json     — Time series for all 7 assets + returns
  dashboard/public/data/sentiment.json  — Daily sentiment with enhanced columns
  dashboard/public/data/countries.json  — Country-level impact data for 3D globe
  dashboard/public/data/meta.json       — Country metadata (roles, regions)

Run after fetch_real_data.py and fetch_sentiment_data.py.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
DASHBOARD_DATA = os.path.join(PROJECT_ROOT, "dashboard", "public", "data")

MARKET_CSV = os.path.join(DATA_DIR, "raw_market_data.csv")
SENTIMENT_CSV = os.path.join(DATA_DIR, "sentiment_scores.csv")
COUNTRY_JSON_SRC = os.path.join(DATA_DIR, "country_sentiment.json")


# Oil-relevant country metadata (mirrored from fetch_sentiment_data.py)
OIL_COUNTRIES = {
    "SAU": {"name": "Saudi Arabia",  "role": "producer",  "region": "Middle East",      "lat": 23.8859,  "lon": 45.0792},
    "IRN": {"name": "Iran",          "role": "producer",  "region": "Middle East",      "lat": 32.4279,  "lon": 53.6880},
    "IRQ": {"name": "Iraq",          "role": "producer",  "region": "Middle East",      "lat": 33.2232,  "lon": 43.6793},
    "KWT": {"name": "Kuwait",        "role": "producer",  "region": "Middle East",      "lat": 29.3117,  "lon": 47.4818},
    "ARE": {"name": "UAE",           "role": "producer",  "region": "Middle East",      "lat": 23.4241,  "lon": 53.8478},
    "VEN": {"name": "Venezuela",     "role": "producer",  "region": "South America",    "lat": 6.4238,   "lon": -66.5897},
    "NGA": {"name": "Nigeria",       "role": "producer",  "region": "Africa",           "lat": 9.0820,   "lon": 8.6753},
    "LBY": {"name": "Libya",         "role": "producer",  "region": "Africa",           "lat": 26.3351,  "lon": 17.2283},
    "DZA": {"name": "Algeria",       "role": "producer",  "region": "Africa",           "lat": 28.0339,  "lon": 1.6596},
    "GAB": {"name": "Gabon",         "role": "producer",  "region": "Africa",           "lat": -0.8037,  "lon": 11.6094},
    "COG": {"name": "Congo",         "role": "producer",  "region": "Africa",           "lat": -0.2280,  "lon": 15.8277},
    "GNQ": {"name": "Eq. Guinea",    "role": "producer",  "region": "Africa",           "lat": 1.6508,   "lon": 10.2679},
    "USA": {"name": "United States", "role": "producer",  "region": "North America",    "lat": 37.0902,  "lon": -95.7129},
    "RUS": {"name": "Russia",        "role": "producer",  "region": "Europe/Asia",      "lat": 61.5240,  "lon": 105.3188},
    "CAN": {"name": "Canada",        "role": "producer",  "region": "North America",    "lat": 56.1304,  "lon": -106.3468},
    "BRA": {"name": "Brazil",        "role": "producer",  "region": "South America",    "lat": -14.2350, "lon": -51.9253},
    "NOR": {"name": "Norway",        "role": "producer",  "region": "Europe",           "lat": 60.4720,  "lon": 8.4689},
    "GBR": {"name": "United Kingdom","role": "producer",  "region": "Europe",           "lat": 55.3781,  "lon": -3.4360},
    "CHN": {"name": "China",         "role": "importer",  "region": "Asia",             "lat": 35.8617,  "lon": 104.1954},
    "IND": {"name": "India",         "role": "importer",  "region": "Asia",             "lat": 20.5937,  "lon": 78.9629},
    "JPN": {"name": "Japan",         "role": "importer",  "region": "Asia",             "lat": 36.2048,  "lon": 138.2529},
    "KOR": {"name": "South Korea",   "role": "importer",  "region": "Asia",             "lat": 35.9078,  "lon": 127.7669},
    "DEU": {"name": "Germany",       "role": "importer",  "region": "Europe",           "lat": 51.1657,  "lon": 10.4515},
    "FRA": {"name": "France",        "role": "importer",  "region": "Europe",           "lat": 46.2276,  "lon": 2.2137},
    "ITA": {"name": "Italy",         "role": "importer",  "region": "Europe",           "lat": 41.8719,  "lon": 12.5674},
    "ESP": {"name": "Spain",         "role": "importer",  "region": "Europe",           "lat": 40.4637,  "lon": -3.7492},
    "TUR": {"name": "Turkey",        "role": "importer",  "region": "Europe/Asia",      "lat": 38.9637,  "lon": 35.2433},
    "TWN": {"name": "Taiwan",        "role": "importer",  "region": "Asia",             "lat": 23.6978,  "lon": 120.9605},
    "OMN": {"name": "Oman",          "role": "producer",  "region": "Middle East",      "lat": 21.4735,  "lon": 55.9754},
    "YEM": {"name": "Yemen",         "role": "transit",   "region": "Middle East",      "lat": 15.5527,  "lon": 48.5164},
    "EGY": {"name": "Egypt",         "role": "transit",   "region": "Middle East",      "lat": 26.8206,  "lon": 30.8025},
    "PAK": {"name": "Pakistan",      "role": "importer",  "region": "South Asia",       "lat": 30.3753,  "lon": 69.3451},
    "SGP": {"name": "Singapore",     "role": "transit",   "region": "Southeast Asia",   "lat": 1.3521,   "lon": 103.8198},
    "PAN": {"name": "Panama",        "role": "transit",   "region": "Central America",  "lat": 8.5380,   "lon": -80.7821},
    "UKR": {"name": "Ukraine",       "role": "transit",   "region": "Europe",           "lat": 48.3794,  "lon": 31.1656},
    "ISR": {"name": "Israel",        "role": "other",     "region": "Middle East",      "lat": 31.0461,  "lon": 34.8516},
}


def convert_market_data():
    """Convert raw_market_data.csv → market.json"""
    print("  Converting market data...")

    if not os.path.exists(MARKET_CSV):
        print(f"    ✗ Not found: {MARKET_CSV}")
        return False

    df = pd.read_csv(MARKET_CSV, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    # Compute log returns
    price_cols = [c for c in df.columns if c != "Date"]
    for col in price_cols:
        safe = df[col].replace(0, np.nan)
        df[f"{col}_Ret"] = np.log(safe / safe.shift(1))

    df.dropna(inplace=True)

    # Convert to JSON-serializable format
    result = {
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "series": {},
    }

    for col in df.columns:
        if col == "Date":
            continue
        result["series"][col] = [
            round(v, 6) if not np.isnan(v) else None
            for v in df[col].tolist()
        ]

    # Add metadata
    result["meta"] = {
        "assets": ["Brent", "WTI", "SP500", "VIX", "Defense_ETF", "EURUSD", "Gold"],
        "start": df["Date"].min().strftime("%Y-%m-%d"),
        "end": df["Date"].max().strftime("%Y-%m-%d"),
        "count": len(df),
    }

    output_path = os.path.join(DASHBOARD_DATA, "market.json")
    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    print(f"    ✓ market.json: {len(df)} rows, {len(result['series'])} series")
    return True


def convert_sentiment_data():
    """Convert sentiment_scores.csv → sentiment.json"""
    print("  Converting sentiment data...")

    if not os.path.exists(SENTIMENT_CSV):
        print(f"    ✗ Not found: {SENTIMENT_CSV}")
        return False

    df = pd.read_csv(SENTIMENT_CSV, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    result = {
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "series": {},
    }

    for col in df.columns:
        if col == "Date":
            continue
        values = df[col].tolist()
        result["series"][col] = [
            round(v, 6) if pd.notna(v) else None for v in values
        ]

    result["meta"] = {
        "start": df["Date"].min().strftime("%Y-%m-%d"),
        "end": df["Date"].max().strftime("%Y-%m-%d"),
        "count": len(df),
        "columns": [c for c in df.columns if c != "Date"],
    }

    output_path = os.path.join(DASHBOARD_DATA, "sentiment.json")
    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    print(f"    ✓ sentiment.json: {len(df)} rows")
    return True


def convert_country_data():
    """Convert country_sentiment.json → countries.json (optimized for globe)"""
    print("  Converting country data...")

    if not os.path.exists(COUNTRY_JSON_SRC):
        print(f"    ✗ Not found: {COUNTRY_JSON_SRC}")
        return False

    with open(COUNTRY_JSON_SRC) as f:
        raw = json.load(f)

    # Just copy and add country metadata
    result = {
        "data": raw,
        "meta": OIL_COUNTRIES,
    }

    output_path = os.path.join(DASHBOARD_DATA, "countries.json")
    with open(output_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    print(f"    ✓ countries.json: {len(raw)} dates")
    return True


def generate_meta():
    """Generate meta.json with country information for the globe."""
    print("  Generating meta.json...")

    output_path = os.path.join(DASHBOARD_DATA, "meta.json")
    with open(output_path, "w") as f:
        json.dump({
            "countries": OIL_COUNTRIES,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "2.0",
        }, f, separators=(",", ":"))

    print(f"    ✓ meta.json: {len(OIL_COUNTRIES)} countries")
    return True


def main():
    print("=" * 60)
    print("  DATA BRIDGE: CSV → JSON for Nuxt Dashboard")
    print("=" * 60)

    os.makedirs(DASHBOARD_DATA, exist_ok=True)

    results = {
        "market": convert_market_data(),
        "sentiment": convert_sentiment_data(),
        "countries": convert_country_data(),
        "meta": generate_meta(),
    }

    print("\n" + "=" * 60)
    total = sum(results.values())
    print(f"  Complete: {total}/{len(results)} files generated")
    for name, ok in results.items():
        print(f"    {'✓' if ok else '✗'} {name}.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
