function train_lstm_model(marketData)
% Multivariate LSTM: Brent price prediction using Brent + VIX + Sentiment

fprintf('\nMultivariate LSTM Prediction\n');

%% Data
projectRoot = '/Volumes/T7/DS Projects/Oil Dependency Price Transmission';
sentimentFile = fullfile(projectRoot, 'Data', 'sentiment_scores.csv');

if ~isfile(sentimentFile)
    fprintf('Sentiment file missing -> using univariate LSTM fallback\n');
    train_lstm_univariate(marketData);
    return;
end

opts = detectImportOptions(sentimentFile, 'VariableNamingRule', 'preserve');
opts = setvartype(opts, 'Date', 'datetime');
sentTable = readtable(sentimentFile, opts);
sentTT = table2timetable(sentTable, 'RowTimes', 'Date');

%% 
combined = synchronize(marketData, sentTT, 'intersection');
combined = rmmissing(combined);

fprintf('Combined dataset observations: %d\n', height(combined));

brent = combined.Brent;
vix   = combined.VIX;
sent  = combined.Sentiment_Mean;
dates = combined.Date;

Xall = [brent vix sent];

%% Train/test split
N = size(Xall, 1);
nTrain = floor(0.8*N);

trainData = Xall(1:nTrain, :);
testData  = Xall(nTrain+1:end, :);

% Standardize
mu = mean(trainData, 1);
sig = std(trainData, 0, 1); sig(sig == 0) = 1;

trainStd = (trainData - mu)./sig;
testStd  = (testData - mu)./sig;

%% Sequences
numLags = 10;
[XTrain, YTrain] = create_sequences(trainStd, numLags);
[XTest, YTest]   = create_sequences(testStd, numLags);

fprintf('Train seq: %d | Test seq: %d\n', length(XTrain), length(XTest));

%% LSTM
numFeatures = size(trainStd, 2);

layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(100, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam',...
    'MaxEpochs',50,...
    'GradientThreshold',1,...
    'InitialLearnRate',0.005,...
    'Verbose',0);

%% Train
try
    net = trainNetwork(XTrain, YTrain, layers, options);
catch ME 
    warning('Training failed -> fallback univariate: %s', ME.message);
    train_lstm_univariate(marketData);
    return;
end

%% Predict
YPredStd = predict(net, XTest, 'MiniBatchSize', 1);

YPred   = YPredStd*sig(1) + mu(1);
YActual = YTest*sig(1) + mu(1);

rmse = sqrt(mean((YPred - YActual).^2));
mae  = mean(abs(YPred - YActual));

fprintf('RMSE: %.4f USD | MAE: %.4f USD\n', rmse, mae);


testDates = dates(nTrain + numLags + 1:end);

figure('Name', 'Multivariate LSTM', 'Color', 'w');
plot(dates(nTrain+1:end), brent(nTrain+1:end), 'k--', 'DisplayName', 'Actual'); hold on
plot(testDates, YPred, 'r', 'LineWidth', 1.5, 'DisplayName', 'Prediction');
legend
title(sprintf('Multivariate LSTM Brent Forecast (RMSE %.2f)', rmse))
grid on

fprintf('Multivariate LSTM completed\n');

end

%% Create Sequences
function [X,Y] = create_sequences(data, numLags)

N = size(data,1);
numSamples = N - numLags;

X = cell(numSamples, 1);
Y = zeros(numSamples, 1);

for i=1:numSamples
    X{i} = data(i:i+numLags-1,:)';
    Y(i) = data(i+numLags,1);
end
end

%% Train LSTM Univariate

function train_lstm_univariate(marketData)

fprintf('Running univariate fallback LSTM\n');

data = marketData.Brent;
dates = marketData.Date;

N = numel(data);
nTrain = floor(0.8*N);

train = data(1:nTrain);
test  = data(nTrain+1:end);

mu = mean(train); sig = std(train);

trainStd = (train - mu)/sig;
testStd  = (test - mu)/sig;

numLags = 10;
[XTrain,YTrain] = create_sequences_1d(trainStd, numLags);
[XTest,YTest]   = create_sequences_1d(testStd, numLags);

layers = [
    sequenceInputLayer(1)
    lstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', 'MaxEpochs', 50, 'Verbose', 0);

net = trainNetwork(XTrain, YTrain, layers, options);

YPredStd = predict(net, XTest, 'MiniBatchSize', 1);
YPred = YPredStd*sig + mu;
YActual = YTest*sig  +mu;

rmse = sqrt(mean((YPred-YActual).^2));
fprintf('Univariate LSTM RMSE: %.3f\n', rmse);

testDates = dates(nTrain+numLags+1:end);

figure('Name', 'Univariate LSTM', 'Color', 'w');
plot(dates(nTrain+1:end), test, 'k--'); hold on
plot(testDates, YPred, 'r', 'LineWidth', 1.5);
title(sprintf('Univariate LSTM Forecast RMSE %.2f', rmse))
grid on

end

%% Sequences 1D

function [X,Y] = create_sequences_1d(data, numLags)

N = numel(data);
numSamples = N - numLags;

X = cell(numSamples, 1);
Y = zeros(numSamples, 1);

for i=1:numSamples
    X{i} = data(i:i+numLags-1)';
    Y(i) = data(i+numLags);
end
end


