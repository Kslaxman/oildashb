"""
fetch_sentiment_data.py
========================
Fetches oil-related news from GDELT (Global Database of Events, Language, and Tone)
and computes daily sentiment scores with country-level impact analysis.

Architecture:
  1. GDELT DOC API v2 (Primary) — free, no API key, 300K+ sources, 15-min updates
  2. Adaptive evergreen queries — no hardcoded conflicts, auto-discovers events
  3. Country-level impact extraction from GDELT article metadata
  4. VIX-calibrated historical proxy with geopolitical event injection (2015→present)

Output:
  Data/sentiment_scores.csv       — Daily global sentiment (backward-compatible)
  Data/country_sentiment.json     — Per-country daily impact data (for 3D globe)
"""

import os
import sys
import json
import re
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import requests
import warnings
warnings.filterwarnings("ignore")

# ── Paths 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
OUTPUT_CSV = os.path.join(DATA_DIR, "sentiment_scores.csv")
OUTPUT_COUNTRY_JSON = os.path.join(DATA_DIR, "country_sentiment.json")
MARKET_CSV = os.path.join(DATA_DIR, "raw_market_data.csv")

# ── GDELT API 
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Evergreen queries — these naturally capture whatever oil-related events
# are happening right now, without hardcoding specific conflicts.
# When US-Iran ends and a new conflict starts, zero code changes needed.
EVERGREEN_QUERIES = [
    '"crude oil" OR "oil price"',
    '"OPEC" OR "oil production cut"',
    '"oil sanctions" OR "oil embargo"',
    '"oil supply disruption" OR "oil shortage"',
    '"energy crisis" OR "oil shock"',
    '"petroleum" OR "oil market crash"',
    '"oil war" OR "oil conflict"',
    '"oil pipeline" OR "refinery attack"',
    '"strait" AND "oil"',              # Catches any strait disruption
    '"oil demand" OR "oil recession"',
]

# Oil-relevant countries with metadata for globe visualization
OIL_COUNTRIES = {
    # OPEC Members
    "SAU": {"name": "Saudi Arabia",  "role": "producer",   "region": "Middle East"},
    "IRN": {"name": "Iran",          "role": "producer",   "region": "Middle East"},
    "IRQ": {"name": "Iraq",          "role": "producer",   "region": "Middle East"},
    "KWT": {"name": "Kuwait",        "role": "producer",   "region": "Middle East"},
    "ARE": {"name": "UAE",           "role": "producer",   "region": "Middle East"},
    "VEN": {"name": "Venezuela",     "role": "producer",   "region": "South America"},
    "NGA": {"name": "Nigeria",       "role": "producer",   "region": "Africa"},
    "LBY": {"name": "Libya",         "role": "producer",   "region": "Africa"},
    "DZA": {"name": "Algeria",       "role": "producer",   "region": "Africa"},
    "GAB": {"name": "Gabon",         "role": "producer",   "region": "Africa"},
    "COG": {"name": "Congo",         "role": "producer",   "region": "Africa"},
    "GNQ": {"name": "Eq. Guinea",    "role": "producer",   "region": "Africa"},
    # Major Non-OPEC Producers
    "USA": {"name": "United States", "role": "producer",   "region": "North America"},
    "RUS": {"name": "Russia",        "role": "producer",   "region": "Europe/Asia"},
    "CAN": {"name": "Canada",        "role": "producer",   "region": "North America"},
    "BRA": {"name": "Brazil",        "role": "producer",   "region": "South America"},
    "NOR": {"name": "Norway",        "role": "producer",   "region": "Europe"},
    "GBR": {"name": "United Kingdom","role": "producer",   "region": "Europe"},
    # Major Importers
    "CHN": {"name": "China",         "role": "importer",   "region": "Asia"},
    "IND": {"name": "India",         "role": "importer",   "region": "Asia"},
    "JPN": {"name": "Japan",         "role": "importer",   "region": "Asia"},
    "KOR": {"name": "South Korea",   "role": "importer",   "region": "Asia"},
    "DEU": {"name": "Germany",       "role": "importer",   "region": "Europe"},
    "FRA": {"name": "France",        "role": "importer",   "region": "Europe"},
    "ITA": {"name": "Italy",         "role": "importer",   "region": "Europe"},
    "ESP": {"name": "Spain",         "role": "importer",   "region": "Europe"},
    "TUR": {"name": "Turkey",        "role": "importer",   "region": "Europe/Asia"},
    "TWN": {"name": "Taiwan",        "role": "importer",   "region": "Asia"},
    # Transit / Strategic
    "OMN": {"name": "Oman",          "role": "producer",   "region": "Middle East"},
    "YEM": {"name": "Yemen",         "role": "transit",    "region": "Middle East"},
    "EGY": {"name": "Egypt",         "role": "transit",    "region": "Middle East"},
    "PAK": {"name": "Pakistan",      "role": "importer",   "region": "South Asia"},
    "SGP": {"name": "Singapore",     "role": "transit",    "region": "Southeast Asia"},
    "PAN": {"name": "Panama",        "role": "transit",    "region": "Central America"},
    "UKR": {"name": "Ukraine",       "role": "transit",    "region": "Europe"},
    "ISR": {"name": "Israel",        "role": "other",      "region": "Middle East"},
}

# Country name → ISO3 mapping for text extraction
COUNTRY_NAME_TO_ISO = {}
for iso, info in OIL_COUNTRIES.items():
    COUNTRY_NAME_TO_ISO[info["name"].lower()] = iso
# Add common aliases
COUNTRY_NAME_TO_ISO.update({
    "saudi": "SAU", "iran": "IRN", "iraq": "IRQ", "russia": "RUS",
    "china": "CHN", "india": "IND", "japan": "JPN", "korea": "KOR",
    "germany": "DEU", "france": "FRA", "italy": "ITA", "spain": "ESP",
    "turkey": "TUR", "ukraine": "UKR", "libya": "LBY", "nigeria": "NGA",
    "venezuela": "VEN", "brazil": "BRA", "canada": "CAN", "norway": "NOR",
    "israel": "ISR", "egypt": "EGY", "oman": "OMN", "yemen": "YEM",
    "kuwait": "KWT", "emirates": "ARE", "uae": "ARE", "u.a.e.": "ARE",
    "algeria": "DZA", "singapore": "SGP", "pakistan": "PAK", "panama": "PAN",
    "british": "GBR", "uk": "GBR", "u.k.": "GBR", "u.s.": "USA",
    "america": "USA", "american": "USA", "taiwan": "TWN",
    "gabon": "GAB", "congo": "COG", "equatorial guinea": "GNQ",
    "south korea": "KOR", "south korean": "KOR",
    "saudi arabia": "SAU", "saudi arabian": "SAU",
    "united states": "USA", "united kingdom": "GBR",
    "persian gulf": "IRN", "hormuz": "IRN", "strait of hormuz": "IRN",
    "opec": "SAU",  # Default OPEC mentions to Saudi as leader
})

# ── Known Geopolitical Events (for historical proxy calibration) 
# These inject realistic sentiment spikes into the VIX proxy for pre-GDELT dates
KNOWN_EVENTS = [
    # (start_date, end_date, sentiment_shock, description)
    ("2014-06-01", "2014-12-31", -0.35, "Oil price collapse 2014"),
    ("2015-01-01", "2015-03-31", -0.25, "Oil oversupply / OPEC refuses cut"),
    ("2016-01-01", "2016-02-28", -0.40, "Oil below $30 / market panic"),
    ("2018-10-01", "2018-12-31", -0.20, "Iran sanctions reimposed / oil volatility"),
    ("2019-09-14", "2019-09-30", -0.45, "Saudi Aramco drone attack"),
    ("2020-01-03", "2020-01-15", -0.30, "US-Iran Soleimani assassination"),
    ("2020-03-06", "2020-04-30", -0.55, "Saudi-Russia price war + COVID crash"),
    ("2020-04-20", "2020-04-21", -0.70, "WTI goes negative"),
    ("2021-02-01", "2021-03-31", -0.10, "Texas freeze / refinery shutdowns"),
    ("2021-07-01", "2021-07-18", -0.15, "OPEC+ internal dispute"),
    ("2022-02-24", "2022-06-30", -0.50, "Russia-Ukraine invasion / energy crisis"),
    ("2022-07-01", "2022-09-30", -0.30, "Europe gas crisis / Nord Stream sabotage"),
    ("2023-10-07", "2023-11-30", -0.35, "Israel-Hamas war / Middle East escalation"),
    ("2024-01-01", "2024-02-28", -0.20, "Houthi Red Sea attacks / shipping disruption"),
    ("2024-04-01", "2024-04-30", -0.25, "Iran-Israel tensions escalation"),
    ("2025-12-01", "2026-01-31", -0.15, "Pre-conflict Iran tensions"),
    ("2026-02-28", "2026-03-31", -0.60, "US-Iran military conflict / Hormuz closure"),
    ("2026-04-01", "2026-04-07", -0.40, "Hormuz disruption / ceasefire negotiations"),
    ("2026-04-08", "2026-04-30", -0.20, "Post-ceasefire uncertainty / Hormuz partial reopen"),
]


# GDELT FETCHING (Recent ~3 months — live data)

def _gdelt_request_with_retry(params, label="", max_retries=3):
    """
    Make a GDELT API request with exponential backoff retry on 429/errors.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                print(f"    Rate limited (429). Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            else:
                print(f"    GDELT status {resp.status_code} for: {label[:50]}")
                return None
        except requests.exceptions.JSONDecodeError:
            print(f"    GDELT returned non-JSON for: {label[:50]}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None
        except Exception as e:
            print(f"    GDELT error for '{label[:50]}': {e}")
            return None
    print(f"    GDELT: all {max_retries} retries exhausted for: {label[:50]}")
    return None


def fetch_gdelt_articles(query, timespan="3m", max_records=250):
    """
    Fetch articles from GDELT DOC API for a single query.
    Returns list of article dicts with title, url, tone, date, source country.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_records,
        "format": "json",
        "TIMESPAN": timespan,
        "sort": "DateDesc",
    }

    data = _gdelt_request_with_retry(params, label=query)
    if data is None:
        return []
    return data.get("articles", [])


def fetch_gdelt_tone_timeline(query, timespan="3m"):
    """
    Fetch daily average tone timeline from GDELT.
    Returns dict of {date_str: avg_tone}.
    """
    params = {
        "query": query,
        "mode": "TimelineTone",
        "format": "json",
        "TIMESPAN": timespan,
        "TIMELINESMOOTH": 1,
    }

    data = _gdelt_request_with_retry(params, label=f"tone:{query}")
    if data is None:
        return {}

    timeline = {}
    if "timeline" in data and len(data["timeline"]) > 0:
        series = data["timeline"][0].get("data", [])
        for point in series:
            date_str = point.get("date", "")
            if date_str and len(date_str) >= 8:
                ds = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                tone_val = point.get("value", 0)
                timeline[ds] = tone_val
    return timeline


def fetch_gdelt_volume_timeline(query, timespan="3m"):
    """
    Fetch daily article volume timeline from GDELT.
    Returns dict of {date_str: article_count}.
    """
    params = {
        "query": query,
        "mode": "TimelineVolRaw",
        "format": "json",
        "TIMESPAN": timespan,
        "TIMELINESMOOTH": 1,
    }

    data = _gdelt_request_with_retry(params, label=f"vol:{query}")
    if data is None:
        return {}

    timeline = {}
    if "timeline" in data and len(data["timeline"]) > 0:
        series = data["timeline"][0].get("data", [])
        for point in series:
            date_str = point.get("date", "")
            if date_str and len(date_str) >= 8:
                ds = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                vol = point.get("value", 0)
                timeline[ds] = vol
    return timeline


def extract_countries_from_text(text):
    """
    Extract oil-relevant country mentions from article title/description.
    Returns list of ISO3 codes found.
    """
    text_lower = text.lower()
    found = set()

    for name, iso in COUNTRY_NAME_TO_ISO.items():
        # Word boundary check to avoid partial matches
        if re.search(r'\b' + re.escape(name) + r'\b', text_lower):
            found.add(iso)

    return list(found)


def rescale_gdelt_tone(tone):
    """
    Rescale GDELT tone (-100 to +100) to [-1, +1] range.
    In practice GDELT tone rarely exceeds ±15 for news articles.
    We use a soft scaling centered on the typical range.
    """
    # Empirical: most GDELT article tones fall between -10 and +10
    return np.clip(tone / 10.0, -1.0, 1.0)


def fetch_all_gdelt_data():
    """
    Master function: fetch articles + timelines for all evergreen queries.
    Returns:
      - daily_sentiment: {date: {tones: [], volumes: int}}
      - country_data: {date: {iso: {tones: [], count: int, snippets: []}}}
    """
    print("\n  Fetching GDELT data across all evergreen queries...")

    daily_sentiment = defaultdict(lambda: {"tones": [], "volumes": 0})
    country_data = defaultdict(lambda: defaultdict(
        lambda: {"tones": [], "count": 0, "snippets": []}
    ))

    for i, query in enumerate(EVERGREEN_QUERIES):
        print(f"    [{i+1}/{len(EVERGREEN_QUERIES)}] {query[:60]}...")

        # Fetch articles (gives us titles for country extraction + per-article tone)
        articles = fetch_gdelt_articles(query, timespan="3m", max_records=250)

        for art in articles:
            title = art.get("title", "") or ""
            url = art.get("url", "") or ""
            tone = art.get("tone", 0) or 0
            date_raw = art.get("seendate", "") or ""
            domain = art.get("domain", "") or ""
            source_country = art.get("sourcecountry", "") or ""

            if not date_raw or len(date_raw) < 8:
                continue

            # Parse date
            date_str = date_raw[:10]  # YYYY-MM-DD or YYYYMMDD
            if "T" in date_str:
                date_str = date_str[:10]
            elif len(date_str) == 8 and "-" not in date_str:
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            # Rescale tone
            scaled_tone = rescale_gdelt_tone(tone)

            # Aggregate daily
            daily_sentiment[date_str]["tones"].append(scaled_tone)
            daily_sentiment[date_str]["volumes"] += 1

            # Extract countries from title
            countries = extract_countries_from_text(title)
            for iso in countries:
                country_data[date_str][iso]["tones"].append(scaled_tone)
                country_data[date_str][iso]["count"] += 1
                if len(country_data[date_str][iso]["snippets"]) < 3:
                    country_data[date_str][iso]["snippets"].append(title[:120])

        # Fetch tone + volume timelines as backup / gap filler
        tone_tl = fetch_gdelt_tone_timeline(query, timespan="3m")
        vol_tl = fetch_gdelt_volume_timeline(query, timespan="3m")

        for ds, tone_val in tone_tl.items():
            scaled = rescale_gdelt_tone(tone_val)
            # Only add if we don't already have article-level data for this date
            if ds not in daily_sentiment or len(daily_sentiment[ds]["tones"]) == 0:
                daily_sentiment[ds]["tones"].append(scaled)

        for ds, vol_val in vol_tl.items():
            daily_sentiment[ds]["volumes"] = max(
                daily_sentiment[ds]["volumes"], int(vol_val)
            )

        # Rate limit: be very gentle with GDELT free API
        time.sleep(5.0)

    print(f"  GDELT: fetched data for {len(daily_sentiment)} unique dates")
    return daily_sentiment, country_data


def process_gdelt_results(daily_sentiment, country_data):
    """
    Convert raw GDELT aggregates into DataFrames.
    Returns:
      - df_daily: DataFrame with Date, Sentiment_Mean, etc.
      - country_json: dict for country_sentiment.json
    """
    # ── Daily global sentiment ──
    records = []
    for date_str, data in sorted(daily_sentiment.items()):
        tones = data["tones"]
        volume = data["volumes"]

        if not tones:
            continue

        records.append({
            "Date": date_str,
            "Sentiment_Mean": float(np.mean(tones)),
            "Sentiment_Std": float(np.std(tones)) if len(tones) > 1 else 0.0,
            "Headline_Count": max(volume, len(tones)),
            "Tone_GDELT": float(np.mean(tones)),
            "Geopolitical_Risk_Flag": 1 if np.mean(tones) < -0.3 else 0,
        })

    df_daily = pd.DataFrame(records)
    if not df_daily.empty:
        df_daily["Date"] = pd.to_datetime(df_daily["Date"])
        df_daily.sort_values("Date", inplace=True)
        df_daily.reset_index(drop=True, inplace=True)

    # ── Country-level JSON ──
    country_json = {}
    for date_str, countries in sorted(country_data.items()):
        day_data = {}
        for iso, info in countries.items():
            if not info["tones"]:
                continue
            avg_tone = float(np.mean(info["tones"]))
            # Generate a reason from the top snippet
            reason = info["snippets"][0] if info["snippets"] else "Oil market coverage"
            day_data[iso] = {
                "tone": round(avg_tone, 4),
                "volume": info["count"],
                "reason": reason,
            }
        if day_data:
            country_json[date_str] = day_data

    return df_daily, country_json


# HISTORICAL PROXY (2015 → GDELT coverage start)

def generate_historical_proxy():
    """
    Generate improved VIX-calibrated sentiment proxy for historical dates.
    Injects known geopolitical event shocks for realism.
    """
    print("\n  Generating VIX-calibrated historical proxy...")

    if not os.path.exists(MARKET_CSV):
        print(f"  Error: Market data not found at {MARKET_CSV}")
        print("  Run fetch_real_data.py first.")
        sys.exit(1)

    market = pd.read_csv(MARKET_CSV, parse_dates=["Date"])

    if "VIX" not in market.columns:
        print("  Error: VIX column not found in market data.")
        sys.exit(1)

    vix = market[["Date", "VIX"]].copy()
    vix.dropna(inplace=True)

    # ── Base sentiment from VIX ──
    vix_min = vix["VIX"].quantile(0.05)
    vix_max = vix["VIX"].quantile(0.95)
    vix_norm = (vix["VIX"].clip(vix_min, vix_max) - vix_min) / (vix_max - vix_min)

    # Invert: low VIX → positive, high VIX → negative
    # Map to [-0.8, 0.8] instead of [-1, 1] to leave room for event injection
    sentiment_base = 0.8 - 1.6 * vix_norm

    # ── Add realistic noise (heteroscedastic — more noise during volatility) ──
    np.random.seed(42)
    noise_scale = 0.03 + 0.05 * vix_norm  # More noise when VIX is high
    noise = np.random.normal(0, noise_scale)
    sentiment = sentiment_base + noise

    # ── Inject known geopolitical events ──
    for start, end, shock, desc in KNOWN_EVENTS:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        mask = (vix["Date"] >= start_dt) & (vix["Date"] <= end_dt)

        if mask.any():
            n_days = mask.sum()
            # Create a shock curve: sharp drop at start, gradual recovery
            t = np.linspace(0, 1, n_days)
            shock_curve = shock * np.exp(-2.0 * t)  # Exponential decay
            # Add some randomness
            shock_noise = np.random.normal(0, abs(shock) * 0.15, n_days)
            sentiment.loc[mask] = sentiment.loc[mask] + shock_curve + shock_noise
            print(f"    Injected event: {desc} ({start} → {end}, shock={shock})")

    sentiment = np.clip(sentiment, -1, 1)

    # ── Compute realistic headline counts (correlated with VIX) ──
    # Higher VIX = more news coverage
    base_count = 15 + 35 * vix_norm  # Range 15-50
    count_noise = np.random.poisson(5, len(base_count))
    headline_counts = (base_count + count_noise).astype(int)

    # ── Compute sentiment std (higher during events) ──
    sentiment_std = 0.05 + 0.15 * vix_norm
    std_noise = np.random.uniform(-0.02, 0.02, len(sentiment_std))
    sentiment_std = np.clip(sentiment_std + std_noise, 0.01, 0.5)

    df = pd.DataFrame({
        "Date": vix["Date"],
        "Sentiment_Mean": sentiment.values,
        "Sentiment_Std": sentiment_std.values,
        "Headline_Count": headline_counts.values,
        "Tone_GDELT": sentiment.values,  # Same as Sentiment_Mean for proxy
        "Geopolitical_Risk_Flag": (sentiment.values < -0.3).astype(int),
    })

    # ── Generate historical country-level impact ──
    country_json = generate_historical_country_data(df, vix)

    return df, country_json


def generate_historical_country_data(df, vix_df):
    """
    Generate estimated country-level impact data for historical dates.
    Based on known events and VIX-correlated country exposure.
    """
    country_json = {}

    # Define which countries are affected during known events
    event_country_map = {
        "Oil price collapse 2014": ["SAU", "RUS", "IRN", "IRQ", "NGA", "VEN", "USA", "CAN", "NOR"],
        "Oil oversupply / OPEC refuses cut": ["SAU", "IRN", "IRQ", "USA", "RUS"],
        "Oil below $30 / market panic": ["SAU", "RUS", "VEN", "NGA", "IRQ", "USA", "CAN", "BRA"],
        "Iran sanctions reimposed": ["IRN", "USA", "CHN", "IND", "TUR", "KOR", "JPN"],
        "Saudi Aramco drone attack": ["SAU", "IRN", "YEM", "USA", "ARE", "KWT"],
        "US-Iran Soleimani assassination": ["IRN", "USA", "IRQ", "ISR", "SAU"],
        "Saudi-Russia price war + COVID crash": ["SAU", "RUS", "USA", "CAN", "NGA", "VEN", "BRA", "NOR"],
        "WTI goes negative": ["USA", "CAN", "SAU", "RUS"],
        "Texas freeze / refinery shutdowns": ["USA"],
        "OPEC+ internal dispute": ["SAU", "ARE", "RUS"],
        "Russia-Ukraine invasion / energy crisis": ["RUS", "UKR", "DEU", "FRA", "ITA", "GBR", "USA", "SAU"],
        "Europe gas crisis / Nord Stream sabotage": ["RUS", "DEU", "FRA", "ITA", "NOR", "GBR"],
        "Israel-Hamas war / Middle East escalation": ["ISR", "IRN", "SAU", "USA", "EGY", "YEM"],
        "Houthi Red Sea attacks / shipping disruption": ["YEM", "SAU", "EGY", "USA", "IRN", "GBR", "SGP"],
        "Iran-Israel tensions escalation": ["IRN", "ISR", "USA", "SAU", "IRQ"],
        "Pre-conflict Iran tensions": ["IRN", "USA", "ISR", "SAU"],
        "US-Iran military conflict / Hormuz closure": ["IRN", "USA", "ISR", "SAU", "IRQ", "KWT", "ARE", "OMN", "CHN", "IND", "JPN", "KOR"],
        "Hormuz disruption / ceasefire negotiations": ["IRN", "USA", "SAU", "ARE", "KWT", "OMN"],
        "Post-ceasefire uncertainty / Hormuz partial reopen": ["IRN", "USA", "SAU", "CHN", "IND"],
    }

    for start, end, shock, desc in KNOWN_EVENTS:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        affected_countries = event_country_map.get(desc, [])

        if not affected_countries:
            continue

        mask = (df["Date"] >= start_dt) & (df["Date"] <= end_dt)
        event_dates = df.loc[mask, "Date"]

        for _, date in event_dates.items():
            date_str = date.strftime("%Y-%m-%d")
            if date_str not in country_json:
                country_json[date_str] = {}

            for iso in affected_countries:
                if iso not in OIL_COUNTRIES:
                    continue
                # Country-specific tone with some variation
                base_tone = shock * (0.7 + 0.6 * np.random.random())
                # Decay over time
                days_in = (date - start_dt).days
                total_days = (end_dt - start_dt).days + 1
                decay = np.exp(-1.5 * days_in / total_days)
                tone = np.clip(base_tone * decay, -1, 1)

                country_json[date_str][iso] = {
                    "tone": round(float(tone), 4),
                    "volume": int(15 + 50 * abs(tone) * np.random.random()),
                    "reason": desc,
                }

    return country_json


def merge_and_save(df_gdelt, df_historical, country_gdelt, country_historical):
    """
    Merge GDELT recent data with historical proxy.
    GDELT data takes priority for overlapping dates.
    """
    print("\n  Merging historical + GDELT data...")

    if df_gdelt is not None and not df_gdelt.empty:
        gdelt_min_date = df_gdelt["Date"].min()
        print(f"    GDELT covers: {df_gdelt['Date'].min().date()} → {df_gdelt['Date'].max().date()}")

        # Historical = everything before GDELT coverage
        df_hist_trimmed = df_historical[df_historical["Date"] < gdelt_min_date].copy()
        print(f"    Historical proxy: {df_hist_trimmed['Date'].min().date()} → {df_hist_trimmed['Date'].max().date()}")

        # Merge
        df_merged = pd.concat([df_hist_trimmed, df_gdelt], ignore_index=True)
    else:
        print("    No GDELT data available — using historical proxy only")
        df_merged = df_historical.copy()

    # Sort and deduplicate
    df_merged.sort_values("Date", inplace=True)
    df_merged.drop_duplicates(subset="Date", keep="last", inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # Format dates for CSV
    df_merged["Date"] = pd.to_datetime(df_merged["Date"]).dt.strftime("%Y-%m-%d")

    # Save CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✓ Saved {len(df_merged)} days of sentiment → {OUTPUT_CSV}")

    # ── Merge country JSON ──
    country_merged = country_historical.copy()
    # GDELT country data overwrites historical for overlapping dates
    for date_str, countries in country_gdelt.items():
        country_merged[date_str] = countries

    # Save country JSON
    with open(OUTPUT_COUNTRY_JSON, "w") as f:
        json.dump(country_merged, f, indent=None, separators=(",", ":"))
    print(f"  ✓ Saved country impact data ({len(country_merged)} dates) → {OUTPUT_COUNTRY_JSON}")

    return df_merged


def main():
    print("=" * 70)
    print("  OIL GEOPOLITICAL SENTIMENT PIPELINE")
    print("  GDELT-Powered · Adaptive Queries · Country-Level Impact")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 1: Historical proxy (2015 → present) ──
    print("\n[1/3] Generating historical sentiment proxy...")
    df_historical, country_historical = generate_historical_proxy()
    print(f"  Historical: {len(df_historical)} days generated")

    # ── Step 2: GDELT recent data (~last 3 months) ──
    print("\n[2/3] Fetching live GDELT sentiment data...")
    try:
        daily_sentiment, country_data = fetch_all_gdelt_data()
        df_gdelt, country_gdelt = process_gdelt_results(daily_sentiment, country_data)
        print(f"  GDELT: {len(df_gdelt)} days of live data")
    except Exception as e:
        print(f"  GDELT fetch failed: {e}")
        print("  Falling back to historical proxy only")
        df_gdelt = pd.DataFrame()
        country_gdelt = {}

    # ── Step 3: Merge & save ──
    print("\n[3/3] Merging and saving...")
    df_final = merge_and_save(df_gdelt, df_historical, country_gdelt, country_historical)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Date range: {df_final['Date'].iloc[0]} → {df_final['Date'].iloc[-1]}")
    print(f"  Total days: {len(df_final)}")

    if "Geopolitical_Risk_Flag" in df_final.columns:
        risk_days = df_final["Geopolitical_Risk_Flag"].sum()
        print(f"  High-risk days: {int(risk_days)} ({risk_days/len(df_final)*100:.1f}%)")

    print(f"\n  Last 5 rows:")
    print(df_final.tail(5).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
