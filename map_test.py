import json
import pandas as pd
import plotly.express as px
from Scripts.generate_api_data import OIL_COUNTRIES

with open("Data/country_sentiment.json") as f:
    raw = json.load(f)

rows = []
for date_str, c_data in raw.items():
    year = date_str[:4]
    for iso, info in c_data.items():
        rows.append({"Date": date_str, "Year": year, "ISO": iso, "Tone": info.get("tone", 0), "Volume": info.get("volume", 0), "Reason": info.get("reason", "Unknown")})

df = pd.DataFrame(rows)

# Get the most significant event per year per country
idx = df.groupby(["Year", "ISO"])["Volume"].idxmax()
df_yearly = df.loc[idx].copy()
df_yearly["Country"] = df_yearly["ISO"].map(lambda x: OIL_COUNTRIES.get(x, {}).get("name", x))
df_yearly["Crisis Severity"] = df_yearly["Volume"] * df_yearly["Tone"].abs()

fig = px.scatter_geo(
    df_yearly,
    locations="ISO",
    size="Crisis Severity",
    color="Tone",
    hover_name="Country",
    hover_data={"Reason": True, "Tone": False, "ISO": False, "Crisis Severity": False},
    animation_frame="Year",
    projection="orthographic",
    color_continuous_scale="RdYlGn"
)
fig.update_geos(
    showcountries=True, countrycolor="#333333",
    showcoastlines=True, coastlinecolor="#333333",
    showland=True, landcolor="#1e1e1e",
    showocean=True, oceancolor="#000000",
    visible=False
)
fig.update_layout(height=600, margin={"r": 0, "t": 0, "l": 0, "b": 0}, template="plotly_dark")
print("Test complete.")
