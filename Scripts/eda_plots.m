function eda_plots(marketData)
% EDA_PLOTS
% Multi-asset exploratory analysis:
% Brent, WTI, SP500, VIX, Defense, EURUSD, Gold

%% PRICE TRENDS
figure('Name', 'Global Market Trends', 'Color', 'w');
tiledlayout(4,1)

% - Oil Prices
nexttile
plot(marketData.Date, marketData.Brent, 'r', 'LineWidth', 1.5); hold on
plot(marketData.Date, marketData.WTI, 'k--', 'LineWidth', 1.2)
ylabel('USD')
title('Oil Prices: Brent vs WTI')
legend('Brent', 'WTI', 'Location', 'best')
grid on
xline(datetime(2022, 2, 24), '--b', 'Ukraine War')

% - Equity vs Defense
nexttile
yyaxis left
plot(marketData.Date, marketData.SP500, 'b', 'LineWidth', 1.5)
ylabel('S&P500')

yyaxis right
plot(marketData.Date, marketData.Defense_ETF, 'g', 'LineWidth', 1.5)
ylabel('Defense ETF')
title('Equity vs Defense Sector')
grid on

% - Gold vs Oil
nexttile
yyaxis left
plot(marketData.Date, marketData.Gold, 'Color', [0.85 0.65 0.13], 'LineWidth', 1.5)
ylabel('Gold')

yyaxis right
plot(marketData.Date, marketData.Brent, 'r', 'LineWidth', 1.2)
ylabel('Brent')
title('Safe Haven: Gold vs Oil')
grid on
xline(datetime(2022, 2, 24), '--b', 'War')

% - Currency & Fear
nexttile
yyaxis left
plot(marketData.Date, marketData.EURUSD, 'Color', [0 0.5 0.5], 'LineWidth', 1.5)
ylabel('EUR/USD')

yyaxis right
area(marketData.Date, marketData.VIX, 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', [0.5 0.5 0.5])
ylabel('VIX')
title('Currency Stress & Market Fear')
grid on

%% ROLLING CORRELATIONS
figure('Name', 'Rolling Correlations', 'Color', 'w');
tiledlayout(3, 1)

window = 30;

% Brent vs SP500
nexttile

function rc = rolling_corr(x,y,w)

rc = NaN(length(x),1);

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

rc1 = rolling_corr(marketData.Brent_Ret, marketData.SP500_Ret, window);
plot(marketData.Date, rc1, 'b', 'LineWidth', 1.5); hold on
yline(0,'k--')
title('30-Day Corr: Brent vs S&P500')
grid on
xline(datetime(2022,2,24), '--r')

% Brent vs Gold
nexttile
rc2 = rolling_corr(marketData.Brent_Ret, marketData.Gold_Ret, window);
plot(marketData.Date, rc2, 'Color', [0.85 0.65 0.13], 'LineWidth', 1.5); hold on
yline(0, 'k--')
title('30-Day Corr: Brent vs Gold')
grid on
xline(datetime(2022,2,24), '--r')

% Brent vs EURUSD
nexttile
rc3 = rolling_corr(marketData.Brent_Ret, marketData.EURUSD_Ret, window);
plot(marketData.Date, rc3, 'Color', [0 0.5 0.5], 'LineWidth', 1.5); hold on
yline(0,'k--')
title('30-Day Corr: Brent vs EURUSD')
grid on
xline(datetime(2022,2,24), '--r')

%% STATIONARITY TEST
fprintf('\n Augmented Dickey-Fuller Test \n');

vars = {'Brent', 'Brent_Ret', 'SP500', 'SP500_Ret', 'Gold', 'Gold_Ret', 'EURUSD', 'EURUSD_Ret'};

for i = 1:length(vars)

    v = vars{i};
    data = marketData.(v);
    data = data(~isnan(data));

    try
        [h,p] = adftest(data);

        if h == 1
            result = 'STATIONARY';
        else
            result = 'NON-STATIONARY';
        end

        fprintf('%-12s | p=%.4f | %s\n',v,p,result);

    catch
        fprintf('%-12s | ADF unavailable\n',v);
    end
end


end
