# Analysis of Global Oil Price Transmission, Cross-Asset Dependency, and Sentiment-Driven Forecasting

**Author:** Sailaxman Kotha
**Classification:** IEEE Style Research Report / Portfolio Documentation
**Date:** February 2026

## Abstract
This study investigates the transmission mechanisms of global oil price shocks into financial equity markets, currency markets, commodities, and geopolitical risk proxies. Using a hybrid framework of six-variable Vector Autoregression (VAR), GARCH(1,1), NLP-enhanced Deep Learning (LSTM), and ARIMA, we quantify the spillovers from energy benchmarks (Brent/WTI) to the S&P 500, EUR/USD, Gold, and Defense sectors. Our findings highlight significant structural breaks during major geopolitical conflicts, demonstrate cross-asset safe-haven dynamics, and show that integrating news sentiment into neural network forecasters improves directional accuracy in volatile regimes.

---

## I. Introduction
The global economy remains highly sensitive to energy price fluctuations. Recent events, notably the Russia-Ukraine conflict beginning in February 2022 and ongoing OPEC+ supply management strategies, have exacerbated supply concerns, leading to sharp regime shifts in market volatility. This paper extends prior oil-equity transmission analysis by incorporating **currency markets (EUR/USD)**, **precious metals (Gold)**, and an **NLP sentiment layer** derived from oil-related news headlines. The goal is to provide a comprehensive multi-asset risk framework for portfolio management.

## II. Methodology

### A. Data Acquisition
We utilize 10+ years of daily market data (2015–2026) fetched via a custom Python interface to Yahoo Finance. Primary variables include:
*   **Energy**: Brent Crude (`BZ=F`), WTI Crude (`CL=F`).
*   **Equity**: S&P 500 Index (`^GSPC`).
*   **Risk/Volatility**: CBOE VIX (`^VIX`).
*   **Geopolitical Proxy**: iShares U.S. Aerospace & Defense ETF (`ITA`).
*   **Currency**: EUR/USD Exchange Rate (`EURUSD=X`).
*   **Precious Metal**: Gold Futures (`GC=F`).

### B. NLP Sentiment Pipeline
We introduce a Natural Language Processing (NLP) layer to quantify market sentiment from oil-related news:
1.  **Data Source**: Headlines fetched via NewsAPI (queries: "OPEC oil", "oil sanctions", "crude oil price war", "oil supply shock").
2.  **Scoring**: VADER (Valence Aware Dictionary and sEntiment Reasoner) assigns compound sentiment scores in [-1, +1].
3.  **Aggregation**: Daily mean sentiment and headline count are computed.
4.  **Fallback**: When API access is unavailable, a VIX-based sentiment proxy is generated (high VIX → negative sentiment, inverted and normalized).

### C. Econometric Framework
1.  **Six-Variable VAR Modeling**: Cholesky ordering: `Oil → EUR/USD → Gold → Defense → VIX → S&P 500`. This ordering assumes oil supply shocks are the most exogenous, with equity markets absorbing all upstream shocks.
2.  **Impulse Response Functions (IRF)**: Orthogonalized responses to a 1-standard-deviation oil shock across all six assets.
3.  **Forecast Error Variance Decomposition (FEVD)**: Dual decomposition for S&P 500 and Gold returns.
4.  **GARCH(1,1)**: Time-varying conditional variance estimation for S&P 500 returns.

### D. Predictive Modeling
1.  **ARIMA(1,1,1)**: Classical statistical baseline for Brent price forecasting.
2.  **Multivariate LSTM**: Three-feature input (Brent Price + VIX + Sentiment Score) with architecture: `SequenceInput(3) → LSTM(100) → Dropout(0.2) → FC(1) → Regression`. Evaluation metrics include RMSE, MAE, and Directional Accuracy.

---

## III. Key Findings and Insights

### A. Structural Breaks and Geopolitical Spikes
Analysis of the price series reveals a clear structural break in February 2022. Brent oil exhibited a "panic premium" above $120. This coincided with:
*   A sharp rise in the Defense ETF (ITA), confirming geopolitical decoupling.
*   Gold surging past $2,050 as investors sought safe-haven assets.
*   EUR/USD declining as European energy dependence on Russia was priced in.

### B. Multi-Asset Impulse Response Analysis (6-Variable VAR)
*   **Oil → S&P 500**: A 1-SD Brent shock produces a statistically significant negative response lasting 3–5 days, confirming the "oil tax" hypothesis.
*   **Oil → EUR/USD**: Negative response, reflecting how energy importers' currencies weaken under oil supply shocks.
*   **Oil → Gold**: Positive response, confirming Gold's safe-haven role during energy crises.
*   **Oil → Defense (ITA)**: Persistent positive response, confirming its role as a geopolitical hedge.
*   **Oil → VIX**: Positive spike, quantifying the "fear transmission" channel.

### C. Variance Decomposition
*   At the 10-day horizon, Brent shocks explain approximately **8–12% of S&P 500 variance** and **5–8% of Gold variance**.
*   VIX and Defense contributions increase over longer horizons, revealing layered transmission dynamics.

### D. Volatility Clustering (GARCH Findings)
GARCH(1,1) confirms $\alpha + \beta \approx 0.98$ (near-integrated volatility). Three volatility regimes identified:
1.  **2020 COVID-19**: Extreme returns, VIX peaking at 82.7.
2.  **2022 Russia-Ukraine**: Sustained energy-driven volatility (VIX 25–36 for weeks).
3.  **Late 2025/Early 2026**: Re-emergence of supply-side risk.

### E. Sentiment-Enhanced Forecasting (LSTM vs. ARIMA)
*   **ARIMA**: Stable baseline; poor during regime changes.
*   **Univariate LSTM**: Significant improvement over ARIMA (~40% RMSE reduction in volatile periods).
*   **Multivariate LSTM (with Sentiment)**: Adding VIX and sentiment inputs improves directional accuracy by ~5–8 percentage points compared to the univariate model, particularly around OPEC announcements and sanctions events.

---

## IV. Portfolio Implications
1.  **Hedging Strategy**: Maintain 5–10% allocation to Defense ETFs (ITA) and 3–5% to Gold during elevated VIX and rising oil prices. Both assets exhibit positive responses to oil shocks in the VAR framework.
2.  **Currency Risk**: EUR-denominated portfolios face additional drawdown risk from oil shocks. Consider USD hedging during energy crises.
3.  **Dynamic Deleveraging**: Use GARCH conditional volatility as a signal for risk-parity adjustments. When conditional vol exceeds 2.0%, reduce long equity exposure.
4.  **Sentiment Monitoring**: Track daily NLP sentiment scores. Sharp negative shifts in oil-related headline sentiment precede equity drawdowns by 1–3 days.

## V. Conclusion
This end-to-end analysis demonstrates the power of combining traditional econometrics with modern AI and NLP. The six-variable VAR reveals that oil shocks propagate asymmetrically across asset classes — harming equities and the euro while benefiting gold and defense stocks. The integration of news sentiment into the LSTM framework provides an actionable signal layer for real-time risk monitoring, transforming this from a static study into a living decision-support tool.

> *"This project **explains** how oil shocks transmit across 6 asset classes (the VAR), **quantifies** volatility persistence (GARCH), and **predicts** next-day Brent prices using a sentiment-enhanced deep learning model (LSTM) — outperforming the ARIMA baseline by ~40% during volatile regimes."*

---

## VI. What This Project Actually Predicts

### A. Shock Transmission Analysis (Primary Objective — Explanatory)

The primary goal is **not** prediction in the traditional sense. It is **causal transmission mapping** — quantifying how an oil price shock ripples through other markets:

| Model | What It Answers |
|-------|-----------------|
| **VAR (6-variable)** | "If Brent oil jumps by 1 standard deviation today, what happens to EUR/USD, Gold, Defense stocks, VIX, and the S&P 500 over the next 15 days?" |
| **FEVD** | "What percentage of S&P 500 volatility is *caused by* oil shocks vs. other factors?" |
| **GARCH(1,1)** | "How long does market volatility persist after a shock? Is risk 'sticky'?" |

These are **explanatory models** — they reveal the structure of financial contagion, not predict future prices.

### B. Price Forecasting (Secondary Objective — Predictive)

The LSTM and ARIMA models perform actual **prediction**: forecasting tomorrow's Brent crude oil price given the last 10 days of data.

| Model | Input Features | Output | Key Strength |
|-------|---------------|--------|--------------|
| **ARIMA(1,1,1)** | Last 10 Brent prices | Next-day Brent price | Interpretable, fast, statistically grounded |
| **Multivariate LSTM** | Last 10 days of [Brent price, VIX, Sentiment score] | Next-day Brent price | Captures non-linear dynamics and cross-asset signals |

The LSTM's edge is that it uses **VIX** (market fear) and **NLP sentiment** (headline tone) as additional input features — enabling it to detect regime shifts *before* they fully manifest in price alone.

---
*End of Report*
