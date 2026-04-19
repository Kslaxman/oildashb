import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_market_data.csv")

TICKERS = {
    "Brent": "BZ=F",
    "WTI": "CL=F",
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "Defense_ETF": "ITA",
    "EURUSD": "EURUSD=X",
    "Gold": "GC=F"
}


def download_with_retry(tickers, retries=3):
    for i in range(retries):
        try:
            data = yf.download(
                list(tickers.values()),
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
                threads=True
            )
            if not data.empty:
                return data
        except Exception as e:
            print(f"Download failed: {e}")
    raise RuntimeError("Failed to download data from Yahoo Finance")


def clean_price_table(raw):
    """Extract close prices safely regardless of Yahoo format"""
    # MultiIndex case
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.levels[0]:
            prices = raw["Adj Close"]
        else:
            prices = raw["Close"]
    else:
        prices = raw

    # Rename columns ticker
    ticker_to_name = {v: k for k, v in TICKERS.items()}
    prices = prices.rename(columns=ticker_to_name)

    # Remove timezone
    prices.index = prices.index.tz_localize(None)

    # fill missing days
    prices = prices.ffill()

    # Remove non-positive values
    prices = prices.where(prices > 0)

    prices.dropna(inplace=True)

    return prices


def save_to_csv(prices):
    prices = prices.copy()
    prices.reset_index(inplace=True)

    prices.rename(columns={"index": "Date"}, inplace=True)

    prices["Date"] = prices["Date"].dt.strftime("%Y-%m-%d")

    os.makedirs(DATA_DIR, exist_ok=True)
    prices.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(prices.head())


def fetch_data():
    print(f"\nFetching data: {START_DATE} → {END_DATE}")

    raw = download_with_retry(TICKERS)
    prices = clean_price_table(raw)
    save_to_csv(prices)


if __name__ == "__main__":
    fetch_data()
