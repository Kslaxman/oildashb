function run_garch_volatility(marketData)
% GARCH (Generalized Auto Regressive Conditional Heteroskedasticity) VOLATILITY
% Estimates GARCH(1,1) model to analyze market risk and volatility clustering
%

fprintf('GARCH Volatility\n');

%% Extract S&P500 returns
ret   = marketData.SP500_Ret;
dates = marketData.Date;

%% Remove missing values
idx = ~isnan(ret);
ret   = ret(idx);
dates = dates(idx);

ret = ret * 100;

%% Specify GARCH(1,1) with constant mean
Mdl = garch(1,1);
fprintf('Estimating GARCH(1,1) on S&P 500 Returns...\n');

% Estimate parameters
[EstMdl, EstParamCov, logL, info] = estimate(Mdl, ret, 'Display', 'off');
disp('Estimated GARCH Model:')
disp(EstMdl)

%% Infer conditional variance
v = infer(EstMdl, ret);
conditionalVol = sqrt(v);

stdRes = ret ./ conditionalVol; % standardized residuals
squareStdResiduals = stdRes .^ 2

figure;
subplot(2, 1, 1);
autocorr(stdRes);
title('ACF: S&P 500 Returns')

subplot(2, 1, 2);
autocorr(squareStdResiduals);
title('ACF: S&P 500 Squared Returns')

%% Annualize volatility
annualizedVol = conditionalVol * sqrt(252); % trading days

%% Plot results
figure('Name', 'GARCH Volatility Analysis', 'Color', 'w');
subplot(2, 1, 1)
plot(dates, ret, 'Color', [0.4 0.4 0.4])
title('S&P 500 Daily Returns (%)')
ylabel('Return (%)')
grid on

subplot(2, 1, 2)
plot(dates, annualizedVol, 'r', 'LineWidth', 1.2)
title('Estimated Annualized Volatility - GARCH(1,1)')
ylabel('Volatility (%)')
grid on
hold on

% Major crisis (Russia invasion)
xline(datetime(2022, 2, 24), '--b', 'Russia-Ukraine War');

fprintf('GARCH Analysis Completed. Volatility clustering identified.\n');

end
