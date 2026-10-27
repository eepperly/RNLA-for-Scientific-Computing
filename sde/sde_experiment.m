%% Spectral density estimation on the transverse-field Ising model
%
% A is the transverse-field Ising Hamiltonian on an open chain of L spins at
% the critical point J = h = 1, shifted and scaled so its spectrum lies in
% [-1,1].  n = 2^L, and the full spectrum comes in closed form from the
% free-fermion solution (TFIM_EIGENVALUES), so the ground truth costs an
% L-by-L SVD rather than a dense eigendecomposition of the 2^L-by-2^L matrix.
%
% Three estimators of the spectral measure mu_A (unit mass on each eigenvalue,
% total mass n):
%
%   SLQ    Stochastic Lanczos quadrature.  m probes, t Lanczos steps
%          each, giving a Gaussian rule exact for degree < 2t.
%   KPM    Kernel polynomial method.  Jackson-damped Chebyshev series
%          built from stochastic moment estimates.
%   CMR    Chebyshev moment regression.  Fits a
%          nonnegative measure of mass n to the moments, weighted by 1/k.
%
% The three are compared at equal numbers of Chebyshev moments K.  KPM and CMR
% each consume m*K/2 matrix-vector products via the doubling trick; SLQ is run
% with t = ceil((K+1)/2) steps, which delivers 2t-1 >= K exact moments for m*t
% products, so the three sit at essentially equal cost.
%
% Top figure:    W_1 error, and tr(f(A)) error for f(u) = e^{-u}, against K.
% Bottom figure: the true eigenvalue histogram against all three estimates.

L      = 16;                 % spins; n = 2^L
Ks     = 2.^(3:10);          % numbers of Chebyshev moments
m      = 5;                  % stochastic probes
trials = 30;                 % repetitions, for the median and 10/90 bands
Kshow  = 128;                % moment count used for the density comparison
nbin   = 120;                % bins for the density comparison

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'));

%% Hamiltonian and its exact spectrum

J = 1;
h = 1;
H   = tfim(L,J,h);
n   = size(H,1);
lam = tfim_eigenvalues(L,J,h);          % closed form; no eig, nothing to cache

% Shift and scale so the spectrum sits strictly inside [-1,1].
lo  = lam(1);
hi  = lam(end);
ctr = (hi + lo)/2;
rad = (hi - lo)/2 / 0.99;

A = (H - ctr*speye(n)) / rad;
d = (lam - ctr) / rad;                      % exact scaled eigenvalues

fprintf('n = %d, scaled spectrum [%.6f, %.6f]\n', n, d(1), d(end));

fun      = @(u) exp(-u);
tr_exact = sum(fun(d));
fprintf('tr(f(A)) exact = %.8e  for f(u) = e^{-u}\n', tr_exact);

% Common grid for the Wasserstein-1 integrals, and the true CDF on it.
% Named wgrid, not grid: a variable called grid would shadow the grid()
% function used to turn on the axes grids below.
wgrid = linspace(-1, 1, 4001)';
Ftrue = measure_cdf(wgrid, d, ones(n,1));

%% Sweep over the number of moments

%          name                          colour     mark style  max K
methods = { ...
    'SLQ',                         '#648FFF', 'o', '-',  Inf; ...
    'KPM',                         '#DC267F', 's', '--', Inf; ...
    'Chebyshev moment regression', '#009E73', '^', '-.', 128};

% SLQ is run without reorthogonalization of the Lanczos basis, the cheap
% O(n t) form.  On this matrix it tracks the fully reorthogonalized version to
% plotting accuracy in W_1, so the O(n t^2) projection is not worth its cost
% here.  slq() still takes reorth as an optional argument if you want to check
% that again on a different spectrum.
%
% CMR stops at K = 128.  Its regression grid is q = ceil(K^1.5), so the convex
% program grows as K^1.5 variables -- 1448 at K = 128, but 11585 at K = 512 --
% and the solve stops being the cheap part of the method long before the
% accuracy stops improving.  SLQ and KPM cost O(K) work per probe and carry on.
% Entries past a method's ceiling are left NaN and simply not drawn.
nmeth = size(methods,1);
nK    = numel(Ks);

W1  = zeros(nmeth, nK, 3);
TRE = zeros(nmeth, nK, 3);

wall = tic;
for j = 1:nK
    K = Ks(j);
    t = ceil((K+1)/2);                      % Lanczos steps matching K moments
    w1 = zeros(trials, nmeth);
    tr = zeros(trials, nmeth);

    for s = 1:trials
        % SLQ
        [nd, wt] = slq(A, n, m, t, false);
        w1(s,1)  = trapz(wgrid, abs(Ftrue - measure_cdf(wgrid, nd, wt)));
        tr(s,1)  = sum(wt .* fun(nd));

        % One set of Chebyshev moments, shared by KPM and CMR
        c = cheb_moments(A, n, m, K);

        % KPM
        [Fk, trk] = kpm_theta(c, n, wgrid, fun);
        w1(s,2)   = trapz(wgrid, abs(Ftrue - Fk));
        tr(s,2)   = trk;

        % Chebyshev moment regression, up to its moment ceiling
        if K <= methods{3,5}
            [xg, wg] = cheb_moment_regression(c, n);
            w1(s,3)  = trapz(wgrid, abs(Ftrue - measure_cdf(wgrid, xg, wg)));
            tr(s,3)  = sum(wg .* fun(xg));
        else
            w1(s,3)  = NaN;
            tr(s,3)  = NaN;
        end
    end

    for i = 1:nmeth
        W1(i,j,:)  = quantile(w1(:,i), [0.10 0.50 0.90]);
        TRE(i,j,:) = quantile(abs(tr(:,i) - tr_exact)/abs(tr_exact), [0.10 0.50 0.90]);
    end
    fprintf('  K = %3d done, %.1f s elapsed\n', K, toc(wall));
end
fprintf('Sweep finished in %.1f s\n', toc(wall));

%% One representative run at K = Kshow for the density comparison

edges  = linspace(-1, 1, nbin+1)';
ctrs   = (edges(1:end-1) + edges(2:end))/2;
bwidth = edges(2) - edges(1);

dens_true = histcounts(d, edges)' / bwidth;

t = ceil((Kshow+1)/2);
[nd, wt]  = slq(A, n, m, t, false);
dens_slq  = bin_measure(nd, wt, edges) / bwidth;

c         = cheb_moments(A, n, m, Kshow);
dens_kpm  = kpm(c, n, ctrs);
[xg, wg]  = cheb_moment_regression(c, n);
dens_cmr  = bin_measure(xg, wg, edges) / bwidth;

%% Plot

% Pixel geometry, shared width, so the two figures line up when dropped into
% LaTeX at a common \includegraphics width.
fig_w     = 1750;           % shared width, so the two line up in LaTeX
top_h     = 700;            % two panels, titles, legend strip
bot_h     = 620;            % one wide panel, title, legend strip
ax_left   = [150 1005];
ax_width  = 630;
top_bot   = 230;            % axes bottom / height inside the top figure
top_ht    = 370;
bot_bot   = 200;            % axes bottom / height inside the bottom figure
bot_ht    = 330;
bot_width = 1485;

% --- Top figure: W_1 and trace error against K ---
fig1 = figure('Units','pixels','Position',[100 100 fig_w top_h], ...
              'PaperPositionMode','auto');

ax1 = axes('Parent',fig1,'Units','pixels', ...
           'Position',[ax_left(1) top_bot ax_width top_ht]);
hold(ax1,'on')
for i = 1:nmeth
    v = ~isnan(W1(i,:,2));          % drop points past this method's ceiling
    plot_shaded(Ks(v), W1(i,v,2)/n, W1(i,v,1)/n, W1(i,v,3)/n, ...
                methods{i,2}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,3}, 'LineStyle', methods{i,4});
end
set(ax1,'XScale','log','YScale','log')
xlabel(ax1,'Number of moments $K$','Interpreter','latex')
ylabel(ax1,'$\mathrm{W}_1(\mu_{\mbox{\boldmath $A$}},\hat\mu)\,/\,n$','Interpreter','latex')
title(ax1,'Wasserstein-1 error','Interpreter','latex')
xlim(ax1,[min(Ks) max(Ks)]); grid(ax1,'on')

ax2 = axes('Parent',fig1,'Units','pixels', ...
           'Position',[ax_left(2) top_bot ax_width top_ht]);
hold(ax2,'on')
for i = 1:nmeth
    v = ~isnan(TRE(i,:,2));
    plot_shaded(Ks(v), TRE(i,v,2), TRE(i,v,1), TRE(i,v,3), ...
                methods{i,2}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,3}, 'LineStyle', methods{i,4});
end
set(ax2,'XScale','log','YScale','log')
xlabel(ax2,'Number of moments $K$','Interpreter','latex')
ylabel(ax2,'$|\hat{\mathrm{tr}}_f - \mathrm{tr}(f(\mbox{\boldmath $A$}))| / |\mathrm{tr}(f(\mbox{\boldmath $A$}))|$', ...
       'Interpreter','latex')
title(ax2,'Spectral sum error, $f(u) = e^{-u}$','Interpreter','latex')
xlim(ax2,[min(Ks) max(Ks)]); grid(ax2,'on')

lines = flipud(findobj(ax2,'Type','Line'));
lgd = legend(lines, methods(:,1), 'Interpreter','latex', ...
             'NumColumns', 3, 'Units','pixels');
lgd.Position(1) = max(0, (fig_w - lgd.Position(3))/2);
lgd.Position(2) = 40;

ansave(fig1, fullfile(here,'..','..','figures','sde_tfim_error'));

% --- Bottom figure: one wide panel comparing densities ---
fig2 = figure('Units','pixels','Position',[100 100 fig_w bot_h], ...
              'PaperPositionMode','auto');
ax3 = axes('Parent',fig2,'Units','pixels', ...
           'Position',[ax_left(1) bot_bot bot_width bot_ht]);
hold(ax3,'on')
% The true spectrum, SLQ and CMR are all discrete: a mass on each eigenvalue,
% and an atomic measure for the two estimators.  Binning them and joining bin
% centres with a curve would draw a continuous density none of them has, so
% they are drawn as histograms over the shared edges.  KPM is the exception --
% it really does return a continuous density -- so it stays a curve, evaluated
% at the bin centres.  The true spectrum is filled and the two estimators are
% outlined, which keeps three overlapping histograms readable.
hTrue = histogram(ax3, 'BinEdges', edges', 'BinCounts', dens_true', ...
                  'FaceColor',[0.35 0.35 0.35], 'FaceAlpha',0.40, 'EdgeColor','none');
hSlq  = histogram(ax3, 'BinEdges', edges', 'BinCounts', dens_slq', ...
                  'DisplayStyle','stairs', 'EdgeColor', methods{1,2}, 'LineWidth', 1.4);
hCmr  = histogram(ax3, 'BinEdges', edges', 'BinCounts', dens_cmr', ...
                  'DisplayStyle','stairs', 'EdgeColor', methods{3,2}, 'LineWidth', 1.4);
hKpm  = plot(ax3, ctrs, dens_kpm, 'Color', methods{2,2}, 'LineWidth', 1.8, 'LineStyle','--');
xlabel(ax3,'$u$','Interpreter','latex')
ylabel(ax3,'Spectral density','Interpreter','latex')
title(ax3, sprintf(['Spectral density of the transverse-field Ising model, ' ...
      '$L = %d$, $K = %d$, $m = %d$ probes'], L, Kshow, m), 'Interpreter','latex')
xlim(ax3,[-1 1]); grid(ax3,'on')

lgd2 = legend([hTrue hSlq hKpm hCmr], ...
              {'True spectrum', methods{1,1}, methods{2,1}, methods{3,1}}, ...
              'Interpreter','latex', 'NumColumns', 4, 'Units','pixels');
lgd2.Position(1) = max(0, (fig_w - lgd2.Position(3))/2);
lgd2.Position(2) = 40;

ansave(fig2, fullfile(here,'..','..','figures','sde_tfim_density'));

%% Local functions

function [F, trf] = kpm_theta(c, n, qgrid, fun)
%KPM_THETA  KPM cumulative mass on `grid`, and the spectral sum for `fun`.
%   Both integrals are taken in theta with y = cos(theta).  There the
%   Chebyshev weight w(y)dy = (2/pi)dtheta is constant, so the endpoint
%   singularity of w at y = +-1 never appears and plain trapezoid quadrature
%   converges quickly.

K  = numel(c);
Nt = 20001;
th = linspace(0, pi, Nt)';
y  = cos(th);

beta = ((K+2-(1:K)').*cos(pi*(1:K)'/(K+2)) ...
        + sin(pi*(1:K)'/(K+2))*cot(pi/(K+2))) / (K+2);

s    = n/2 * ones(Nt,1);
Tkm1 = ones(Nt,1);
Tk   = y;
for k = 1:K
    s = s + beta(k)*c(k)*Tk;
    Tnew = 2*y.*Tk - Tkm1;
    Tkm1 = Tk;
    Tk   = Tnew;
end

trf = (2/pi) * trapz(th, s .* fun(y));

% theta runs 0 -> pi as y runs 1 -> -1, so mass below y is the tail integral.
tot  = trapz(th, s);
Fth  = (2/pi) * (tot - cumtrapz(th, s));
F    = interp1(th, Fth, acos(max(-1, min(1, qgrid))), 'linear', 'extrap');

end

function b = bin_measure(nodes, weights, edges)
%BIN_MEASURE  Total weight of a discrete measure falling in each bin.
idx = discretize(nodes, edges);
ok  = ~isnan(idx);
b   = accumarray(idx(ok), weights(ok), [numel(edges)-1, 1]);
end
