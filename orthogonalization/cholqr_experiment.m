%% Randomized CholeskyQR against the classical alternatives
%
% Orthogonalize a tall-thin matrix of prescribed condition number four ways and
% measure how well each one does it:
%
%   CholeskyQR       Gram matrix plus Cholesky.  Loses orthogonality like
%                    kappa(A)^2*eps and breaks down entirely near kappa ~ 1e8.
%   CholeskyQR2      The same, twice, with a shifted first pass.  Never breaks
%                    down, and recovers full accuracy while kappa <~ 1e8;
%                    beyond that its orthogonality degrades.
%   RandCholeskyQR   Whiten with a SparseStack sketch, then CholeskyQR.
%   qr               MATLAB's Householder QR, the stable reference.
%
% A = U*diag(s)*V' is m-by-n with s logarithmically spaced from 1 to 1/kappa,
% so kappa(A) is exactly the swept parameter.
%
% Left panel:  loss of orthogonality ||Q'Q - I|| against kappa(A), on a small
%              matrix where accuracy is the only thing being measured.
% Right panel: runtime against n, on tall 5e5-by-n matrices where the cost
%              actually shows.  These are well conditioned (plain randn) so
%              every method succeeds and the timings are comparable.

m      = 2000;                  % rows, for the accuracy sweep
n      = 100;                   % columns, for the accuracy sweep
kappas = 10.^(0:1:16);          % condition numbers to sweep

mt     = 5e5;                   % rows, for the timing sweep
ns     = [10 20 50 100 200 500];   % columns to sweep
trials = 30;                    % repetitions for the accuracy sweep
ttrials = 30;                   % repetitions for the timing sweep

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'), fullfile(here,'..','sketching'));

%          name               factorization              color      mark  style
methods = { ...
    'CholeskyQR',       @(A) CholeskyQR(A),        '#DC267F', 's', '--'; ...
    'CholeskyQR2',      @(A) CholeskyQR2(A),       '#FE6100', 'd', '-.'; ...
    'RandCholeskyQR',   @(A) RandCholeskyQR(A),    '#648FFF', 'o', '-';  ...
    'MATLAB qr',        @(A) qr(A,0),              '#009E73', '^', '-'};
nmeth = size(methods,1);
nkap  = numel(kappas);

orth = nan(nmeth, nkap, 3);
resd = nan(nmeth, nkap, 3);

wall = tic;
for j = 1:nkap
    s = logspace(0, -log10(kappas(j)), n)';

    for i = 1:nmeth
        f  = methods{i,2};
        to = nan(trials,1);
        tr = nan(trials,1);
        for t = 1:trials
            [U, ~] = qr(randn(m,n), 0);
            [V, ~] = qr(randn(n));
            A = (U .* s') * V';

            % A failed Cholesky is the result, not an error: leave the entry
            % NaN so the curve simply stops where the method breaks down.
            try
                [Q, R] = f(A);
                to(t) = norm(Q'*Q - eye(n));
                tr(t) = norm(A - Q*R) / norm(A);
            catch ME
                if ~contains(ME.identifier, 'NotPositiveDefinite')
                    rethrow(ME)
                end
            end
        end
        if any(~isnan(to))
            orth(i,j,:) = quantile(to(~isnan(to)), [0.10 0.50 0.90]);
            resd(i,j,:) = quantile(tr(~isnan(tr)), [0.10 0.50 0.90]);
        end
        fprintf('  kappa=%.0e  %-16s orth %.2e  resid %.2e  (%d/%d succeeded)\n', ...
                kappas(j), methods{i,1}, orth(i,j,2), resd(i,j,2), sum(~isnan(to)), trials);
    end
end
fprintf('Sweep finished in %.1f s\n', toc(wall));

%% Timing sweep: runtime against n on tall 5e5-by-n matrices

nn   = numel(ns);
time = nan(nmeth, nn, 3);

wall = tic;
for j = 1:nn
    A = randn(mt, ns(j));               % well conditioned, so nothing breaks
    for i = 1:nmeth
        f  = methods{i,2};
        tt = zeros(ttrials,1);
        for t = 1:ttrials
            tic;
            [Qt, Rt] = f(A);            %#ok<ASGLU> timed, then dropped
            tt(t) = toc;
            clear Qt Rt
        end
        time(i,j,:) = quantile(tt, [0.10 0.50 0.90]);
        fprintf('  n=%4d  %-16s %7.3f s\n', ns(j), methods{i,1}, time(i,j,2));
    end
    clear A
end
fprintf('Timing sweep finished in %.1f s\n', toc(wall));

%% Plot

% Pixel geometry, so the export has a deterministic size.
% ax_bottom has to cover the tick labels and x label (~110 px) plus the legend
% strip beneath them (~120 px); fig_h then leaves ~95 px above for the titles.
fig_w     = 1750;
fig_h     = 700;
ax_left   = [150 1005];
ax_width  = 630;
ax_bottom = 230;
ax_height = 375;

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h], ...
             'PaperPositionMode','auto');

ax1 = axes('Parent',fig,'Units','pixels', ...
           'Position',[ax_left(1) ax_bottom ax_width ax_height]);
hold(ax1,'on')
for i = 1:nmeth
    plot_shaded(kappas, orth(i,:,2), orth(i,:,1), orth(i,:,3), ...
                methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
plot(kappas, eps*kappas.^2, 'k:',  'LineWidth', 1.2);
plot(kappas, eps*ones(size(kappas)), 'k--', 'LineWidth', 1.2);
set(ax1,'XScale','log','YScale','log')
xlabel(ax1,'$\kappa(\mbox{\boldmath $A$})$','Interpreter','latex')
ylabel(ax1,'$\|\mbox{\boldmath $Q$}^\top\mbox{\boldmath $Q$}-\mbox{\boldmath $I$}\|$', ...
       'Interpreter','latex')
title(ax1,'Loss of orthogonality','Interpreter','latex')
xlim(ax1,[1 1e16]); ylim(ax1,[1e-17 1e2]); grid(ax1,'on')

ax2 = axes('Parent',fig,'Units','pixels', ...
           'Position',[ax_left(2) ax_bottom ax_width ax_height]);
hold(ax2,'on')
for i = 1:nmeth
    plot_shaded(ns, time(i,:,2), time(i,:,1), time(i,:,3), ...
                methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
set(ax2,'XScale','log','YScale','log')
xlabel(ax2,'$n$','Interpreter','latex')
ylabel(ax2,'Runtime (s)','Interpreter','latex')
title(ax2,'Runtime','Interpreter','latex')
xlim(ax2,[min(ns) max(ns)]); grid(ax2,'on')

lines = flipud(findobj(ax1,'Type','Line'));
lgd = legend(lines, [methods(:,1); {'$\kappa^2\varepsilon_{\rm mach}$'}; {'$\varepsilon_{\rm mach}$'}], ...
             'Interpreter','latex', 'NumColumns', 6, 'Units','pixels');
lgd.Position(1) = max(0, (fig_w - lgd.Position(3))/2);
lgd.Position(2) = 35;

ansave(fig, fullfile(here,'..','..','figures','cholqr_orthogonality'));
