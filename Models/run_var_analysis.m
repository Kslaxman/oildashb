function run_var_analysis(marketData)
% VAR () ANALYSIS
% 6-Variable VAR Shock Transmission Model
%
% Order (Cholesky identification):
% Oil -> Currency -> Gold -> Defense -> Fear -> Equity

fprintf('\nMulti-Asset VAR Modeling (6 Variables)\n');

varNames = {'Brent_Ret', 'EURUSD_Ret', 'Gold_Ret', 'Defense_ETF_Ret', 'VIX_Ret', 'SP500_Ret'};

dataSubset = marketData(:, varNames);
dataSubset = rmmissing(dataSubset);

Y = dataSubset.Variables;
numVars = size(Y, 2); % number of cols

fprintf('Variables: %s\n', strjoin(varNames, ', '));
fprintf('Observations: %d\n', size(Y, 1));

%% LAG ORDER SELECTION
maxLag = 10;
aic = NaN(maxLag, 1);
bic = NaN(maxLag, 1);

for p = 1:maxLag
    try
        Mdl = varm(numVars, p); % vector autoregression model
        [~, ~, ~, info] = estimate(Mdl, Y, 'Display', 'off');
        aic(p) = info.AIC;
        bic(p) = info.BIC;
    catch
        continue
    end
end

[~, bestLagAIC] = min(aic);
[~, bestLagBIC] = min(bic);

fprintf('Optimal Lag (AIC): %d\n', bestLagAIC);
fprintf('Optimal Lag (BIC): %d\n', bestLagBIC);

p_opt = max(1, bestLagBIC);

%% ESTIMATE VAR
fprintf('\nEstimating VAR(%d)...\n', p_opt);
Mdl = varm(numVars, p_opt);
Mdl.SeriesNames = varNames;

EstMdl = estimate(Mdl, Y);
summarize(EstMdl);

%% IMPULSE RESPONSE FUNCTIONS
numPeriods = 15;

try
    [Y_IRF, Y_L, Y_U] = irf(EstMdl, 'NumPeriods', numPeriods);
catch
    try
        [Y_IRF, Y_L, Y_U] = irf(EstMdl, numPeriods);
    catch
        [Y_IRF, Y_L, Y_U] = irf(EstMdl);
        numPeriods = size(Y_IRF, 1);
    end
end

% Indices
idx_Oil     = find(strcmp(varNames, 'Brent_Ret'));
idx_EURUSD  = find(strcmp(varNames, 'EURUSD_Ret'));
idx_Gold    = find(strcmp(varNames, 'Gold_Ret'));
idx_Defense = find(strcmp(varNames, 'Defense_ETF_Ret'));
idx_VIX     = find(strcmp(varNames, 'VIX_Ret'));
idx_SP500   = find(strcmp(varNames, 'SP500_Ret'));

figure('Name', 'IRF: Oil Shock Transmission', 'Color', 'w');
tiledlayout(2, 3)

nexttile; plot_irf_single(Y_IRF, Y_L,Y_U, idx_Oil, idx_Oil, varNames, numPeriods)
nexttile; plot_irf_single(Y_IRF, Y_L, Y_U, idx_EURUSD, idx_Oil, varNames, numPeriods)
nexttile; plot_irf_single(Y_IRF, Y_L, Y_U, idx_Gold, idx_Oil, varNames, numPeriods)
nexttile; plot_irf_single(Y_IRF, Y_L, Y_U, idx_Defense, idx_Oil, varNames, numPeriods)
nexttile; plot_irf_single(Y_IRF, Y_L, Y_U, idx_VIX, idx_Oil, varNames, numPeriods)
nexttile; plot_irf_single(Y_IRF, Y_L, Y_U, idx_SP500, idx_Oil, varNames, numPeriods)

sgtitle('Impulse Response: 1 S.D. Oil Shock -> All Assets')

%% FORECAST ERROR VARIANCE DECOMPOSITION
try
    fevd_vals = fevd(EstMdl, 'NumPeriods', numPeriods);
catch
    try
        fevd_vals = fevd(EstMdl, numPeriods);
    catch
        fevd_vals = fevd(EstMdl);
    end
end

figure('Name', 'FEVD Analysis', 'Color', 'w');
tiledlayout(1, 2)

% S&P500 FEVD
nexttile
sp500_fevd = squeeze(fevd_vals(:, idx_SP500, :));
area(1:numPeriods, sp500_fevd)
legend(varNames, 'Location', 'bestoutside', 'FontSize', 7)
title('Variance Decomposition: S&P 500')
ylabel('Variance Share'); 
xlabel('Periods'); 
ylim([0 1]); 
grid on

% Gold FEVD
nexttile
gold_fevd = squeeze(fevd_vals(:, idx_Gold, :));
area(1:numPeriods, gold_fevd)
legend(varNames, 'Location', 'bestoutside', 'FontSize', 7)
title('Variance Decomposition: Gold')
ylabel('Variance Share'); 
xlabel('Periods'); 
ylim([0 1]); 
grid on

fprintf('\nVAR Analysis Completed.\n');
fprintf('Significance: if CI bands do not cross zero.\n');

end


function plot_irf_single(Y, L, U, respIdx, shockIdx, names, n)

t = 0:(n-1);
y = Y(:, respIdx, shockIdx);
l = L(:, respIdx, shockIdx);
u = U(:, respIdx, shockIdx);

hold on
fill([t fliplr(t)], [l' fliplr(u')], [0.9 0.9 0.9], 'EdgeColor', 'none')
plot(t, y, 'b', 'LineWidth', 2)
yline(0, 'k--')
title(sprintf('%s → %s', names{shockIdx}, names{respIdx}))
xlim([0 n-1]); 
grid on
hold off

end
