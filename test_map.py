import json
import pandas as pd
import plotly.express as px
from Scripts.generate_api_data import OIL_COUNTRIES

with open("Data/country_sentiment.json") as f:
    raw = json.load(f)

# Convert to DataFrame
rows = []
for date_str, c_data in raw.items():
    year = date_str[:4]
    for iso, info in c_data.items():
        rows.append({"Date": date_str, "Year": year, "ISO": iso, "Tone": info["tone"]})

df = pd.DataFrame(rows)
df_yearly = df.groupby(["Year", "ISO"])["Tone"].mean().reset_index()

# Merge with OIL_COUNTRIES
df_yearly["Country"] = df_yearly["ISO"].map(lambda x: OIL_COUNTRIES.get(x, {}).get("name", x))

fig = px.choropleth(
    df_yearly,
    locations="ISO",
    color="Tone",
    hover_name="Country",
    animation_frame="Year",
    color_continuous_scale="RdYlGn",
    projection="mercator",
    title="Global Oil Sentiment by Year"
)
fig.update_geos(fitbounds="locations", visible=False)
fig.write_json("test_map_output.json")
print("Done")
