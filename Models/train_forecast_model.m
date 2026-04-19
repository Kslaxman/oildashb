function train_forecast_model(marketData)
% TRAIN_FORECAST_MODEL
% Forecast Brent oil price using ARIMA(1,1,1)

fprintf('ARIMA Forecasting Model\n');

%% Extract data
price = marketData.Brent;
dates = marketData.Date;

price = price(:);
dates = dates(:);

%% Split train/test
n = length(price);
nTrain = floor(0.8*n);

train = price(1:nTrain);
test  = price(nTrain+1:end);
testDates = dates(nTrain+1:end);

%% Fit ARIMA(1,1,1)
model = arima(1,1,1);

fprintf('Estimating ARIMA model...\n');
fitModel = estimate(model,train,'Display','off');

%% Forecast
numSteps = length(test);
forecasted = forecast(fitModel,numSteps,'Y0',train);

%% Evaluate
rmse = sqrt(mean((forecasted - test).^2));
fprintf('Forecast RMSE: %.4f USD\n',rmse);

%% Plot
figure('Name','Oil Price Forecast','Color','w');
plot(dates,price,'k','DisplayName','Actual Price'); 
hold on
plot(testDates,forecasted,'r','LineWidth',1.5,'DisplayName','Forecast');
legend('Location','best')
title(sprintf('Brent Oil ARIMA Forecast (RMSE = %.2f USD)',rmse))
xlabel('Date')
ylabel('Price (USD)')
grid on

end
