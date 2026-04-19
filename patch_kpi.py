import pandas as pd
import numpy as np

df = pd.read_csv("Data/raw_market_data.csv", parse_dates=["Date"]).set_index("Date").sort_index().dropna()
last_30 = df["Brent"].iloc[-30:]
log_prices = np.log(last_30.values)
coeffs = np.polyfit(np.arange(30), log_prices, 1)
tomorrow_forecast = np.exp(np.polyval(coeffs, 30))
print(f"Today: {last_30.iloc[-1]}, Tomorrow Forecast: {tomorrow_forecast}")
