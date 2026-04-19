import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Oil Shock Transmission Dashboard", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef; }
    [data-testid="stSidebar"] { background-color: #f1f3f5; }
</style>
""", unsafe_allow_html=True)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_data():
    market_df = pd.read_csv("Data/raw_market_data.csv", parse_dates=["Date"])
    sent_df = pd.read_csv("Data/sentiment_scores.csv", parse_dates=["Date"])
    
    # Preprocessing
    market_df = market_df.sort_values("Date")
    sent_df = sent_df.sort_values("Date")
    
    # Calculate returns for KPI metrics
    price_cols = [c for c in market_df.columns if c != "Date"]
    for col in price_cols:
        market_df[f"{col}_Ret"] = market_df[col].pct_change()
        
    return market_df, sent_df

market_df, sent_df = load_data()

# --- SIDEBAR NAV ---
st.sidebar.title("🛢️ Global Oil Intelligence")
st.sidebar.markdown("---")
tab_selection = st.sidebar.radio(
    "Select Dashboard View",
    ["📈 Price Trends", "🌍 Multi-Country", "🔗 Correlation", "📊 Volatility", "🧠 Sentiment", "📋 Raw Data"]
)

st.sidebar.markdown("---")
date_range = st.sidebar.date_input(
    "Date Range",
    value=(market_df["Date"].min(), market_df["Date"].max()),
    min_value=market_df["Date"].min(),
    max_value=market_df["Date"].max()
)

# Filter Data
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) if len(date_range) > 1 else pd.to_datetime(date_range[0])
filtered_market = market_df[(market_df["Date"] >= start_date) & (market_df["Date"] <= end_date)]

# --- 6-KPI HEADER ---
st.title("Oil Dependency & Price Shock Transmission")
col1, col2, col3, col4, col5, col6 = st.columns(6)

def format_metric(df, col, is_price=True):
    val = df[col].iloc[-1]
    ret = df[f"{col}_Ret"].iloc[-1] * 100
    prefix = "$" if is_price else ""
    return f"{prefix}{val:,.2f}" if val < 100 else f"{prefix}{val:,.0f}", f"{ret:+.2f}%"

m1_val, m1_delta = format_metric(filtered_market, "Brent")
col1.metric("Brent Crude", m1_val, m1_delta)

m2_val, m2_delta = format_metric(filtered_market, "SP500", is_price=False)
col2.metric("S&P 500", m2_val, m2_delta)

m3_val, m3_delta = format_metric(filtered_market, "Gold")
col3.metric("Gold Futures", m3_val, m3_delta)

m4_val, m4_delta = format_metric(filtered_market, "EURUSD", is_price=False)
col4.metric("EUR/USD", f"{filtered_market['EURUSD'].iloc[-1]:.4f}", m4_delta)

col5.metric("VIX Index", f"{filtered_market['VIX'].iloc[-1]:.1f}")

m6_val, m6_delta = format_metric(filtered_market, "Defense_ETF")
col6.metric("Defense (ITA)", m6_val, m6_delta)

st.markdown("---")

# --- TAB CONTENT ---
if tab_selection == "📈 Price Trends":
    st.subheader("Global Energy & Market Benchmarks")
    
    fig1 = px.line(filtered_market, x="Date", y=["Brent", "WTI"], 
                  title="Oil Prices: Brent vs WTI",
                  color_discrete_map={"Brent": "red", "WTI": "black"})
    fig1.add_vline(x=datetime(2022, 2, 24).timestamp() * 1000, line_dash="dash", line_color="blue", annotation_text="Ukraine War")
    st.plotly_chart(fig1, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=filtered_market["Date"], y=filtered_market["SP500"], name="S&P 500"), secondary_y=False)
        fig2.add_trace(go.Scatter(x=filtered_market["Date"], y=filtered_market["Defense_ETF"], name="Defense ETF", line=dict(dash="dot")), secondary_y=True)
        fig2.update_layout(title_text="Equity vs Defense Sector")
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_b:
        fig3 = px.area(filtered_market, x="Date", y="VIX", title="Market Fear (VIX Index)", color_discrete_sequence=["gray"])
        st.plotly_chart(fig3, use_container_width=True)

elif tab_selection == "🌍 Multi-Country":
    st.subheader("Geopolitical & Currency Transmission")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=filtered_market["Date"], y=filtered_market["EURUSD"], name="EUR/USD", line=dict(color="teal")))
    fig4.update_layout(title="Currency Channel: European Energy Dependence (EUR/USD)")
    st.plotly_chart(fig4, use_container_width=True)
    
    fig5 = px.line(filtered_market, x="Date", y=["Gold", "Brent"], title="Safe Haven Dynamics: Gold vs Brent")
    st.plotly_chart(fig5, use_container_width=True)

elif tab_selection == "🔗 Correlation":
    st.subheader("30-Day Rolling Correlations")
    window = st.sidebar.slider("Correlation Window (Days)", 10, 120, 30)
    
    corr_df = filtered_market.set_index("Date")
    rc1 = corr_df["Brent_Ret"].rolling(window).corr(corr_df["SP500_Ret"])
    rc2 = corr_df["Brent_Ret"].rolling(window).corr(corr_df["Gold_Ret"])
    rc3 = corr_df["Brent_Ret"].rolling(window).corr(corr_df["EURUSD_Ret"])
    
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=corr_df.index, y=rc1, name="Brent ↔ S&P 500"))
    fig_corr.add_trace(go.Scatter(x=corr_df.index, y=rc2, name="Brent ↔ Gold"))
    fig_corr.add_trace(go.Scatter(x=corr_df.index, y=rc3, name="Brent ↔ EUR/USD"))
    fig_corr.update_layout(title=f"{window}-Day Rolling Correlations (Log Returns)")
    st.plotly_chart(fig_corr, use_container_width=True)

elif tab_selection == "📊 Volatility":
    st.subheader("Asset Volatility Clustering")
    v_window = st.sidebar.slider("Volatility Window (Days)", 10, 60, 21)
    
    vol_df = filtered_market.set_index("Date")
    v1 = vol_df["SP500_Ret"].rolling(v_window).std() * np.sqrt(252) * 100
    v2 = vol_df["Brent_Ret"].rolling(v_window).std() * np.sqrt(252) * 100
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=vol_df.index, y=v1, name="S&P 500 Annualized Vol"))
    fig_vol.add_trace(go.Scatter(x=vol_df.index, y=v2, name="Brent Annualized Vol"))
    fig_vol.update_layout(title=f"{v_window}-Day Realized Volatility (%)")
    st.plotly_chart(fig_vol, use_container_width=True)

elif tab_selection == "🧠 Sentiment":
    st.subheader("NLP News Sentiment Engine")
    
    sent_filtered = sent_df[(sent_df["Date"] >= start_date) & (sent_df["Date"] <= end_date)]
    sent_filtered = sent_filtered.set_index("Date")
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        colors = ["green" if v > 0 else "red" for v in sent_filtered["Sentiment_Mean"]]
        fig_s = go.Figure(go.Bar(x=sent_filtered.index, y=sent_filtered["Sentiment_Mean"], marker_color=colors))
        fig_s.update_layout(title="Daily NLP Sentiment Score (VADER)")
        st.plotly_chart(fig_s, use_container_width=True)
        
    with col_s2:
        st.metric("Headline Count", int(sent_filtered["Headline_Count"].sum()))
        st.metric("Avg Sentiment", f"{sent_filtered['Sentiment_Mean'].mean():.2f}")

elif tab_selection == "📋 Raw Data":
    st.subheader("Dataset Explorer")
    data_choice = st.radio("Select Dataset", ["Market Data", "Sentiment Data"], horizontal=True)
    if data_choice == "Market Data":
        st.dataframe(filtered_market)
        st.download_button("Download CSV", filtered_market.to_csv().encode('utf-8'), "market_data.csv", "text/csv")
    else:
        st.dataframe(sent_df)
        st.download_button("Download CSV", sent_df.to_csv().encode('utf-8'), "sentiment_data.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("Oil shock transmission analysis engine. Built with Streamlit and Plotly.")
