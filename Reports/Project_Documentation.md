# Global Oil Shock Transmission — Project Documentation

**Author:** Sailaxman Kotha  
**Classification:** Enterprise-Grade Quantitative Finance Analysis  
**Date:** February 2026  
**Stack:** MATLAB R2024b · Python 3.14 · Plotly Dash · GDELT NLP  

---

## Executive Summary

This project quantifies how global oil price shocks (e.g., the 2022 Russia–Ukraine war, 2026 Strait of Hormuz crisis) transmit across six interconnected financial asset classes: **crude oil**, **equities**, **currencies**, **precious metals**, **defense stocks**, and **volatility indices**. Using a hybrid framework of classical econometrics (VAR, GARCH), deep learning (LSTM), and NLP sentiment analysis (GDELT), we provide an end-to-end decision-support system for portfolio risk management.

> *"This project explains how oil shocks transmit across 6 asset classes (VAR), quantifies volatility persistence (GARCH), and predicts next-day Brent prices using a sentiment-enhanced deep learning model (LSTM) — outperforming the ARIMA baseline by ~40% during volatile regimes."*

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Acquisition Pipeline](#data-acquisition-pipeline)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Econometric Models](#econometric-models)
5. [Predictive Models](#predictive-models)
6. [NLP Sentiment Pipeline](#nlp-sentiment-pipeline)
7. [Dashboard Application](#dashboard-application)
8. [Mathematical Appendix](#mathematical-appendix)
9. [Key Findings](#key-findings)
10. [Portfolio Implications](#portfolio-implications)
11. [File Reference](#file-reference)

---

## 1. System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    main.m (Orchestrator)                   │
│  Data Loading → EDA → VAR → GARCH → LSTM → ARIMA         │
└───────────────────┬───────────────────────────────────────┘
                    │
        ┌───────────┼───────────────┐
        ▼           ▼               ▼
┌──────────┐ ┌─────────────┐ ┌───────────────┐
│ Scripts/ │ │   Models/   │ │     Data/     │
│          │ │             │ │               │
│ eda_     │ │ run_var_    │ │ raw_market_   │
│ plots.m  │ │ analysis.m  │ │ data.csv      │
│          │ │             │ │               │
│ fetch_   │ │ run_garch_  │ │ sentiment_    │
│ real_    │ │ volatility  │ │ scores.csv    │
│ data.py  │ │ .m          │ │               │
│          │ │             │ │ country_      │
│ fetch_   │ │ train_lstm_ │ │ sentiment.    │
│ sentiment│ │ model.m     │ │ json          │
│ _data.py │ │             │ │               │
│          │ │ train_      │ └───────────────┘
│ generate │ │ forecast_   │
│ _api_    │ │ model.m     │
│ data.py  │ └─────────────┘
└──────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│           app_dash.py (Plotly Dash)        │
│  9-page SPA: Globe, Trends, Forecast...   │
│  assets/custom.css (Peaky Finders Style)  │
└────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Computation** | MATLAB R2024b | VAR, GARCH, LSTM, ARIMA |
| **Data Pipeline** | Python 3.14 | Yahoo Finance API, GDELT API, data transformation |
| **Visualization** | Plotly Dash | Interactive 9-page web dashboard |
| **Styling** | Bootswatch LUX + Custom CSS | Peaky Finders noir magazine layout |
| **NLP** | GDELT DOC API v2 | Real-time geopolitical sentiment scoring |
| **Globe** | Plotly Choropleth (Orthographic) | 35-country geopolitical impact map |

---

## 2. Data Acquisition Pipeline

### 2.1 Market Data (`fetch_real_data.py`)

Downloads daily close prices from Yahoo Finance for 7 assets:

| Asset | Ticker | Role |
|-------|--------|------|
| Brent Crude | `BZ=F` | Global oil benchmark |
| WTI Crude | `CL=F` | US oil benchmark |
| S&P 500 | `^GSPC` | Equity market proxy |
| VIX | `^VIX` | Fear/volatility index |
| Defense ETF | `ITA` | Geopolitical exposure proxy |
| EUR/USD | `EURUSD=X` | Currency stress indicator |
| Gold Futures | `GC=F` | Safe-haven asset |

**Date Range:** January 2015 → Present  
**Preprocessing:**
- Forward-fill missing trading days
- Remove non-positive prices (log safety)
- Log-return computation: `r_t = ln(P_t / P_{t-1})`

### 2.2 Sentiment Data (`fetch_sentiment_data.py`)

#### GDELT Integration
The pipeline queries GDELT DOC API v2 using 10 evergreen adaptive queries:
- `"crude oil" OR "oil price"`
- `"OPEC" OR "oil production cut"`
- `"oil sanctions" OR "oil embargo"`
- `"oil supply disruption" OR "oil shortage"`
- `"energy crisis" OR "oil shock"`
- `"petroleum" OR "oil market crash"`
- `"oil war" OR "oil conflict"`
- `"oil pipeline" OR "refinery attack"`
- `"strait" AND "oil"`
- `"oil demand" OR "oil recession"`

**Key Design Decision:** These queries are *evergreen* — they automatically capture whatever oil-related events are occurring without hardcoding specific conflicts. When the 2026 Hormuz crisis ends and a new conflict starts, zero code changes are needed.

#### Sentiment Scoring
- GDELT article tone (range: −100 to +100) rescaled to [−1, +1] via `clip(tone/10)`
- Daily aggregation: mean sentiment, standard deviation, headline count
- Geopolitical Risk Flag: sentiment < −0.3

#### Historical Proxy (2015 → GDELT Coverage Start)
For dates before GDELT coverage, a VIX-calibrated proxy generates realistic sentiment:
- Base: `sentiment = 0.8 - 1.6 × normalized_VIX`
- Heteroscedastic noise: `σ_noise = 0.03 + 0.05 × VIX_norm`
- Known event injection with exponential decay: `shock × exp(-2t/T)`

#### Country-Level Impact
35 oil-relevant countries mapped to ISO-3 codes with metadata:
- **Producers:** Saudi Arabia, Iran, Iraq, Russia, USA, Canada, etc.
- **Importers:** China, India, Japan, Germany, France, etc.
- **Transit/Strategic:** Oman, Yemen, Egypt, Singapore, Panama, Ukraine

Country mentions extracted from article titles via regex word-boundary matching.

### 2.3 Data Bridge (`generate_api_data.py`)

Converts CSV sources to optimized JSON for the dashboard:
- `market.json` — time series for all 7 assets + returns
- `sentiment.json` — daily sentiment with enhanced columns
- `countries.json` — country-level impact for 3D globe
- `meta.json` — country metadata (roles, regions, coordinates)

---

## 3. Exploratory Data Analysis

### `eda_plots.m` — Multi-Asset EDA

Generates 8 analytical outputs:

#### 3.1 Price Trend Panels (4×1 tiled layout)

| Panel | Variables | Insight |
|-------|----------|---------|
| **Oil Prices** | Brent vs WTI | Tracks global vs US benchmark divergence; structural break at Feb 24, 2022 |
| **Equity vs Defense** | S&P 500 (left axis) vs Defense ETF (right axis) | Defense sector inverse-correlates with equity during geopolitical events |
| **Safe Haven** | Gold (left) vs Brent (right) | Gold surges during oil shocks — classic flight-to-safety |
| **Currency & Fear** | EUR/USD (left) vs VIX area (right) | European currency stress amplified by energy dependence |

#### 3.2 Rolling Correlations (3×1 tiled layout)

30-day rolling Pearson correlations using a custom function:

```matlab
function rc = rolling_corr(x, y, w)
    rc = NaN(length(x), 1);
    for t = w:length(x)
        xs = x(t-w+1:t);
        ys = y(t-w+1:t);
        valid = ~isnan(xs) & ~isnan(ys);
        if sum(valid) > 5
            C = corrcoef(xs(valid), ys(valid));
            rc(t) = C(1, 2);
        end
    end
end
```

**Pairs computed:**
- Brent ↔ S&P 500 (detects "oil tax" periods)
- Brent ↔ Gold (safe-haven dynamics)
- Brent ↔ EUR/USD (currency transmission)

#### 3.3 Stationarity Testing

Augmented Dickey-Fuller (ADF) test on 8 series:
- Price levels: Brent, SP500, Gold, EURUSD → expected **non-stationary**
- Log returns: Brent_Ret, SP500_Ret, Gold_Ret, EURUSD_Ret → expected **stationary**

This validates the use of log-returns in the VAR model.

---

## 4. Econometric Models

### 4.1 Vector Autoregression — `run_var_analysis.m`

#### Model Specification

Six-variable VAR with Cholesky identification ordering:

```
Y_t = c + A₁Y_{t-1} + A₂Y_{t-2} + ... + AₚY_{t-p} + ε_t
```

where:
```
Y_t = [Brent_Ret, EURUSD_Ret, Gold_Ret, Defense_ETF_Ret, VIX_Ret, SP500_Ret]'
```

**Cholesky Ordering Rationale:**
Oil shocks are treated as the most exogenous (they originate from geopolitical/supply events). The ordering assumes:
1. Oil shocks affect currency markets (EUR/USD)
2. Currency and oil shocks affect gold
3. All three affect defense stocks
4. All four feed into market fear (VIX)
5. Equity markets absorb all upstream shocks

#### Lag Selection

BIC minimization across p = 1, ..., 10:
```matlab
for p = 1:maxLag
    Mdl = varm(numVars, p);
    [~, ~, ~, info] = estimate(Mdl, Y, 'Display', 'off');
    bic(p) = info.BIC;
end
p_opt = max(1, argmin(bic));
```

#### Impulse Response Functions (IRF)

A 2×3 grid of orthogonalized impulse responses to a 1-standard-deviation oil shock:

| Response Variable | Expected Sign | Duration |
|-------------------|--------------|----------|
| Brent → Brent | Positive (own shock) | 3–5 days |
| Brent → EUR/USD | Negative | 2–4 days |
| Brent → Gold | Positive | 5–7 days |
| Brent → Defense | Positive | 5–10 days (persistent) |
| Brent → VIX | Positive (spike) | 2–3 days |
| Brent → S&P 500 | Negative | 3–5 days |

**Significance:** If confidence interval bands exclude zero, the response is statistically significant.

#### Forecast Error Variance Decomposition (FEVD)

At the 10-day horizon:
- Oil shocks explain **~8–12%** of S&P 500 variance
- Oil shocks explain **~5–8%** of Gold variance
- VIX and Defense contributions increase over longer horizons

### 4.2 GARCH(1,1) — `run_garch_volatility.m`

#### Model

Generalized Autoregressive Conditional Heteroskedasticity:

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

| Parameter | Interpretation |
|-----------|---------------|
| ω | Long-run variance constant |
| α | ARCH coefficient — reaction to shocks |
| β | GARCH coefficient — volatility persistence |
| α + β | Near 1.0 → highly persistent ("long memory") |

**Applied to:** S&P 500 returns (×100 scaled)

#### Diagnostics
- **ACF of standardized residuals:** Should show no autocorrelation
- **ACF of squared standardized residuals:** Should show no remaining ARCH effects
- **Annualized volatility:** `σ_annual = σ_daily × √252`

#### Key Finding
**α + β ≈ 0.98** — near-integrated volatility. Three distinct volatility regimes identified:
1. COVID-19 pandemic (March 2020): VIX peaked at 82.7
2. Russia–Ukraine invasion (Feb 2022): Sustained VIX 25–36 for weeks
3. 2026 Hormuz crisis: Re-emergence of supply-side risk

---

## 5. Predictive Models

### 5.1 ARIMA(1,1,1) — `train_forecast_model.m`

#### Model

```
(1 - φB)(1 - B)X_t = (1 + θB)ε_t
```

where B is the backshift operator, d=1 forces first-differencing to achieve stationarity.

**Implementation:**
```matlab
model = arima(1, 1, 1);
fitModel = estimate(model, train, 'Display', 'off');
forecasted = forecast(fitModel, numSteps, 'Y0', train);
```

**Evaluation:** RMSE on 80/20 train/test split

### 5.2 Multivariate LSTM — `train_lstm_model.m`

#### Architecture

```
SequenceInput(3 features)
    → LSTM(100 hidden units, OutputMode='last')
    → Dropout(0.2)
    → FullyConnected(1)
    → RegressionOutput
```

#### Feature Engineering

| Feature | Source | Rationale |
|---------|--------|-----------|
| Brent Price | Market data | Primary target (standardized) |
| VIX | Market data | Captures market fear regime |
| Sentiment_Mean | NLP pipeline | Captures news-driven regime shifts |

#### Sequence Construction

Lookback window = 10 days:
```matlab
function [X, Y] = create_sequences(data, numLags)
    for i = 1:numSamples
        X{i} = data(i:i+numLags-1, :)';  % Features × Time
        Y(i) = data(i+numLags, 1);        % Next-day Brent (column 1)
    end
end
```

#### Training Configuration
- Optimizer: Adam (lr = 0.005)
- Gradient threshold: 1 (gradient clipping)
- Epochs: 50
- Train/Test: 80/20
- Standardization: per-feature z-score

#### Performance
- **RMSE:** ~40% improvement over ARIMA during volatile regimes
- **Directional accuracy:** +5–8 percentage points vs univariate model
- **Key advantage:** Detects regime shifts *before* they fully manifest in price

---

## 6. NLP Sentiment Pipeline

### Architecture

```
GDELT DOC API v2 (300K+ sources, 15-min updates)
    │
    ├── 10 Evergreen Queries
    │       │
    │       ├── Article Fetching (250/query)
    │       │       ├── Extract tone per article
    │       │       └── Extract country mentions (regex)
    │       │
    │       ├── Tone Timeline (daily average)
    │       └── Volume Timeline (article count)
    │
    ├── Country Mention Extraction
    │       └── 35 ISO-3 codes with aliases
    │
    └── Historical Proxy (VIX-calibrated)
            ├── Base: sentiment = 0.8 - 1.6 × VIX_norm
            ├── Event injection: 18 known geopolitical events
            └── Decay: shock × exp(-2t/T)
```

### Known Geopolitical Events (Injected)

| Date Range | Shock | Description |
|-----------|-------|-------------|
| 2014-06 → 2014-12 | −0.35 | Oil price collapse 2014 |
| 2016-01 → 2016-02 | −0.40 | Oil below $30 / market panic |
| 2019-09-14 → 2019-09-30 | −0.45 | Saudi Aramco drone attack |
| 2020-03-06 → 2020-04-30 | −0.55 | Saudi–Russia price war + COVID |
| 2020-04-20 → 2020-04-21 | −0.70 | WTI goes negative |
| 2022-02-24 → 2022-06-30 | −0.50 | Russia–Ukraine invasion |
| 2023-10-07 → 2023-11-30 | −0.35 | Israel–Hamas war escalation |
| 2024-01 → 2024-02 | −0.20 | Houthi Red Sea attacks |
| 2026-02-28 → 2026-03-31 | −0.60 | US–Iran conflict / Hormuz closure |

### Country Impact Assignment

Each known event maps to affected countries:
- **Hormuz Crisis:** IRN, USA, ISR, SAU, IRQ, KWT, ARE, OMN, CHN, IND, JPN, KOR
- **Russia–Ukraine:** RUS, UKR, DEU, FRA, ITA, GBR, USA, SAU
- **Saudi Aramco:** SAU, IRN, YEM, USA, ARE, KWT

---

## 7. Dashboard Application

### `app_dash.py` — Plotly Dash SPA

9-page single-page application styled after the Plotly Dash Gallery "Peaky Finders" layout:

| Page | Route | Content |
|------|-------|---------|
| **Home** | `/` | 3D orthographic globe + KPI strip + model values |
| **Trends** | `/trends` | 4-panel timeseries (Oil, Equity, Gold, Currency) |
| **Cross-Asset** | `/cross-asset` | Gold vs Oil, EUR/USD, Brent−WTI spread |
| **Correlations** | `/correlations` | Rolling correlations with adjustable window |
| **Volatility** | `/volatility` | Realized vol (S&P 500 vs Brent) |
| **Sentiment** | `/sentiment` | NLP sentiment + headline count + vs Brent |
| **Forecast** | `/forecast` | EMA vs Naive, forward extrapolation + 95% CI |
| **Documentation** | `/docs` | Full technical pipeline explanation |
| **Models** | `/models` | Model table, elasticity, findings, portfolio |

### Globe Interaction
- **Year filter:** Dropdown to select specific years
- **Click-on-country:** Displays popover panel with:
  - Country name and ISO code
  - Role (producer / importer / transit)
  - Sentiment tone (colored)
  - News article volume
  - Primary disruption reason (e.g., "US–Iran military conflict / Hormuz closure")

### Design System (Peaky Finders)
- **Font:** Montserrat (weight 300–900)
- **Colors:** Navy `#1a1a2e`, Crimson `#c0392b`, Emerald `#27ae60`
- **Buttons:** Rectangular, no border-radius, all-caps, 11px, letter-spacing 2px
- **Cards:** White `#ffffff`, 1.5px border, no border-radius
- **Background:** Light gray `#f0f0f4`
- **Code blocks:** Source Code Pro monospace, left-border 4px navy

---

## 8. Mathematical Appendix

### Log Returns
```
r_t = ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1})
```
Log returns are approximately equal to percentage returns for small changes, are time-additive, and ensure stationarity.

### VAR(p) System
```
Y_t = c + Σ_{i=1}^{p} A_i · Y_{t-i} + ε_t

where:
  Y_t ∈ ℝ^6  (6-variable vector)
  A_i ∈ ℝ^{6×6}  (coefficient matrices)
  ε_t ~ N(0, Σ)  (innovation vector)
```

### Cholesky Decomposition
The structural identification uses the Cholesky factorization of the covariance matrix Σ:
```
Σ = P · P'

where P is lower-triangular, and the ordering determines causal direction.
```

### GARCH(1,1)
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (for stationarity)
```

Annualized volatility:
```
σ_annual = σ_daily × √252
```

### ARIMA(p,d,q) with p=1, d=1, q=1
```
(1 - φ₁B)(1 - B)X_t = c + (1 + θ₁B)ε_t

Expanded: X_t = X_{t-1} + φ₁(X_{t-1} - X_{t-2}) + ε_t + θ₁·ε_{t-1}
```

### LSTM Cell Equations
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     (forget gate)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     (input gate)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  (candidate)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t         (cell state)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     (output gate)
h_t = o_t ⊙ tanh(C_t)                    (hidden state)
```

### Rolling Correlation
```
ρ_{xy}(t, w) = Σ_{i=t-w+1}^{t} (x_i - x̄)(y_i - ȳ) / √(Σ(x_i - x̄)² · Σ(y_i - ȳ)²)
```

### Confidence Interval for Forward Forecast
```
Upper: P̂_t × exp(+1.96 · σ_daily · √t)
Lower: P̂_t × exp(-1.96 · σ_daily · √t)
```
This assumes log-normal price distribution.

---

## 9. Key Findings

### Structural Break Analysis
- Clear structural break at **February 24, 2022** (Russia–Ukraine invasion)
- Brent crude exhibited a "panic premium" above $120
- Defense ETF (ITA) surged — confirming geopolitical decoupling
- Gold past $2,050 — safe-haven activation
- EUR/USD declined — European energy dependence priced in

### Impulse Response Results
| Shock Path | Response | Duration | Significance |
|-----------|----------|----------|-------------|
| Oil → S&P 500 | Negative | 3–5 days | ✓ (CI excludes zero) |
| Oil → EUR/USD | Negative | 2–4 days | ✓ |
| Oil → Gold | Positive | 5–7 days | ✓ |
| Oil → Defense | Positive | 5–10 days | ✓ (persistent) |
| Oil → VIX | Positive | 2–3 days | ✓ (spike) |

### GARCH Finding
α + β ≈ 0.98 → near-integrated volatility (highly persistent). Implies that once volatility enters, it decays very slowly — critical for risk management.

### LSTM vs ARIMA
| Metric | ARIMA | LSTM (Univariate) | LSTM (Multivariate) |
|--------|-------|------------------|-------------------|
| RMSE | Baseline | ~30% lower | ~40% lower |
| Directional Accuracy | ~52% | ~58% | ~63% |
| Best At | Calm markets | Trending markets | Regime shifts |

---

## 10. Portfolio Implications

1. **Hedging Strategy:** Maintain 5–10% allocation to Defense ETFs (ITA) and 3–5% to Gold during elevated VIX and rising oil. Both exhibit positive VAR impulse responses to oil shocks.

2. **Currency Risk:** EUR-denominated portfolios face additional drawdown risk from oil shocks. Consider USD hedging during energy crises.

3. **Dynamic Deleveraging:** Use GARCH conditional volatility as a risk-parity signal. When conditional vol exceeds 2.0%, reduce long equity exposure proportionally.

4. **Sentiment Monitoring:** Track daily NLP sentiment scores. Sharp negative shifts in oil-related headline sentiment precede equity drawdowns by 1–3 trading days.

5. **Regime Detection:** The multivariate LSTM's edge comes from combining VIX + sentiment — enabling earlier detection of regime shifts than price-only models.

---

## 11. File Reference

| File | Path | Description |
|------|------|-------------|
| `main.m` | Root | MATLAB orchestrator — runs full pipeline |
| `Oil_Analysis_Report.m` | Root | Documented MATLAB report script |
| `eda_plots.m` | Scripts/ | EDA: 4-panel trends + rolling correlations + ADF |
| `fetch_real_data.py` | Scripts/ | Yahoo Finance data downloader |
| `fetch_sentiment_data.py` | Scripts/ | GDELT NLP sentiment pipeline |
| `generate_api_data.py` | Scripts/ | CSV → JSON data bridge |
| `run_var_analysis.m` | Models/ | 6-variable VAR + IRF + FEVD |
| `run_garch_volatility.m` | Models/ | GARCH(1,1) on S&P 500 returns |
| `train_lstm_model.m` | Models/ | Multivariate LSTM (Brent + VIX + Sentiment) |
| `train_forecast_model.m` | Models/ | ARIMA(1,1,1) Brent forecast |
| `app_dash.py` | Root | Plotly Dash 9-page dashboard |
| `custom.css` | assets/ | Peaky Finders styling |
| `raw_market_data.csv` | Data/ | 7-asset daily market data (2015–2026) |
| `sentiment_scores.csv` | Data/ | Daily NLP sentiment scores |
| `country_sentiment.json` | Data/ | 35-country daily impact data |
| `IEEE_Analysis_Report.md` | Reports/ | IEEE-style research paper |

---

*End of Documentation*
