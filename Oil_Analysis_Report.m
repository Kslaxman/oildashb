%% GLOBAL OIL TRADE DEPENDENCY & PRICE SHOCK TRANSMISSION ANALYSIS
% Author: Sailaxman Kotha
% Date: Feb, 2026
%
% This project studies how global oil price shocks (e.g., the 2022 Russia-Ukraine war)
% transmit to financial markets (S&P 500) and geopolitical risk proxies (Defense ETFs).
%
% Used a multi-stage framework combining:
%   - Econometrics (VAR, GARCH)
%   - Deep Learning (LSTM)

%% 1. INTRODUCTION & OBJECTIVES
% Primary Goal:
%   Quantify the impact of oil supply shocks on portfolio volatility.
%
% Hypothesis:
%   Structural breaks in oil prices (wars, sanctions) create "risk-off"
%   behavior and increase volatility spillovers into equity markets.
%
% Methodology:
%   1) Data acquisition (Yahoo Finance via Python interface)
%   2) Exploratory Data Analysis
%   3) Vector Autoregression (Impulse Response Analysis)
%   4) GARCH(1,1) volatility modeling
%   5) LSTM neural network prediction

clear; clc; close all;

addpath('Scripts');
addpath('Models');

%% 2. DATA ACQUISITION
% Data Sources:
% Brent Crude (BZ=F)      - Global oil benchmark
% WTI Crude (CL=F)        - US oil benchmark
% S&P 500 (^GSPC)         - Equity market risk proxy
% VIX (^VIX)              - Fear index
% ITA ETF                 - Defense/geopolitical exposure proxy

fprintf('STEP 1: Importing data...\n');
marketData = import_data();

disp(head(marketData))

%% 3. STRUCTURAL BREAK VISUALIZATION
% We visualize price dynamics to detect war shocks,
% especially around February 2022.

fprintf('STEP 2: Running EDA plots...\n');
eda_plots(marketData);

%% 4. ECONOMETRIC ANALYSIS — VAR (SHOCK TRANSMISSION)
% Question:
% Does an oil price increase cause equity market decline?
%
% We simulate a 1-standard deviation oil shock using impulse responses.

fprintf('STEP 3: Running VAR analysis...\n');
run_var_analysis(marketData);

%% 5. RISK MODELING — GARCH
% Financial volatility clusters over time.
% Large shocks are followed by large shocks.
%
% We estimate time-varying market risk.

fprintf('STEP 4: Running GARCH volatility model...\n');
run_garch_volatility(marketData);

%% 6. PREDICTIVE MODELING — LSTM
% Deep learning model captures nonlinear oil price dynamics
% and regime shifts.

fprintf('STEP 5: Running LSTM prediction...\n');
train_lstm_model(marketData);

%% 7. CONCLUSION & PORTFOLIO IMPLICATIONS
% Hedging:
%   Defense ETFs may hedge equity downside during geopolitical conflicts.
%
% Risk Management:
%   GARCH results imply persistent volatility regimes -> dynamic deleveraging.
%
% Future Work:
%   Add NLP news sentiment (Sanctions, OPEC announcements) to improve forecasting accuracy.

