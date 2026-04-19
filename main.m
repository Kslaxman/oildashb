%% MAIN_OIL_ANALYSIS
% Global Oil Trade Dependency & Price Shock Transmission Analysis
% Pipeline:
% 1. Data Loading
% 2. EDA
% 3. VAR (Oil -> Currency -> Gold -> Defense -> Fear -> Equity)
% 4. GARCH (S&P500 volatility)
% 5. LSTM
% 6. ARIMA forecast

clear; clc; close all;

projectRoot = '/Volumes/T7/DS Projects/Oil Dependency Price Transmission';
cd(projectRoot);

addpath('Scripts');
addpath('Models');

fprintf('GLOBAL OIL SHOCK TRANSMISSION ANALYSIS\n');

%% 1 - LOAD MARKET DATA

fprintf('\nImporting Data\n')

csvFile = fullfile('Data', 'raw_market_data.csv');

if ~isfile(csvFile)
    error('Data file not found: %s\nRun: python3 Scripts/fetch_real_data.py first.', csvFile);
end

opts = detectImportOptions(csvFile, 'VariableNamingRule', 'preserve');
opts = setvartype(opts, 'Date', 'datetime');

rawTable   = readtable(csvFile, opts);
marketData = table2timetable(rawTable, 'RowTimes', 'Date');
marketData = sortrows(marketData);

%% Compute log returns
vars = marketData.Properties.VariableNames;

for i = 1:length(vars)
    name = vars{i};

    if isnumeric(marketData.(name)) && ~endsWith(name, '_Ret')

        prices = marketData.(name);
        prices(prices<=0) = NaN;        % log safety

        logReturns = [NaN; diff(log(prices))];
        marketData.([name '_Ret']) = logReturns;
    end
end

marketData(1,:) = []; % remove NaN row

fprintf('Loaded %d observations, %d variables\n',...
        height(marketData), width(marketData));

%% 2 — EDA
fprintf('\nExploratory Data Analysis (EDA)\n');
eda_plots(marketData);
fprintf('EDA completed\n');

%% 3 — VAR MODEL
fprintf('\nVAR Dependency Modeling (6-variable)\n');
run_var_analysis(marketData);

%% 4 — GARCH MODEL
fprintf('\nGARCH Volatility Modeling\n');
run_garch_volatility(marketData);

%% 5 — PREDICTION MODELS
fprintf('\nPrediction Modeling\n');

% LSTM 
if exist('sequenceInputLayer','file') == 2
    fprintf('Running Multivariate LSTM...\n');
    train_lstm_model(marketData);
else
    fprintf('Deep Learning Toolbox not installed -> skipping LSTM\n');
end

% ARIMA
fprintf('Running ARIMA Forecast...\n');
train_forecast_model(marketData);
