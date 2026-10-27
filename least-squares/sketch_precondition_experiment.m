%% Sketch-and-precondition: initialization, refinement, and runtime
%
% Test problem from RANDOM_LS_PROBLEM, m-by-n with m = 5e5 and n = 200:
%
%   A = U*diag(s)*V',   s logarithmically spaced from 1e-11 to 1,
%
% with U and V random with orthonormal columns, so cond(A) = 1e11, and
% b = A*x + r for a random unit vector x and a residual r orthogonal to
% range(A) with ||r|| = 1e-6, so x is the least-squares solution.
%
% Three variants of sketch-and-precondition (CG on the randomly preconditioned
% normal equations), all from SketchAndPrecondition.m:
%
%   Zero init.              Start from x = 0.
%   Sketch-and-solve init.  Start from the sketch-and-solve solution.
%   + refinement            Sketch-and-solve start, one step of iterative
%                           refinement halfway through the iteration budget, as
%                           in the main text of Epperly, Meier & Nakatsukasa
%                           (CPAM 2025): recompute c = R'\A'*(b - A*x) and restart
%                           CG.  This is SPIR with a fixed iteration count.
%
% The implementation is the QR-based Algorithm 4 of that paper, S*A = Q*R, with
% a sparse embedding with zeta = 4 and d = 12n.  The sketch-and-
% solve baseline uses a smaller sketch, s = 2n.  No column scaling is applied.
% An alternative diagonal-plus-noise test matrix is left commented out below;
% for it, column scaling alone takes cond(A) to ~1.04, since its
% ill-conditioning is almost entirely in the column norms.
%
% Top left:    forward error against iteration.
% Bottom left: Karlson-Walden estimate of the backward error, relative to
%              ||A||_F, with theta = ||A||_F/||b|| as in the paper.  This is the
%              exact KW estimate, computed from an SVD of A, not the sketched one.
% Right:       forward error against runtime for the two variants without
%              refinement, at iteration budgets plot_every, 2*plot_every, ...,
%              iters, with sketch-and-solve and A\b as baselines.  Markers are
%              medians;
%              bars span the 10% to 90% quantiles in both directions.
% A\b is also drawn as a dashed reference level in the left panels.

m          = 5e5;      % rows
n          = 200;      % columns
resid      = 1e-6;     % residual norm of the least-squares solution
iters      = 30;       % iteration budget for the left panels
plot_every = 2;        % reporting frequency in iterations, for all three panels
trials     = 30;       % repetitions, for the median and 10/90 quantiles

% Iteration budgets for the runtime panel: every plot_every iterations, the same
% grid the left panels report on.  iters itself is always included.
ks = unique([plot_every:plot_every:iters, iters]);

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'), fullfile(here,'..','sketching'));

%% Problem

rng(1);
[A,b,x,~,SA,VA] = random_ls_problem(m,n,1e11,resid);

% A = [diag(logspace(0,-12,n)); zeros(m-n,n)] + 1e-14*randn(m,n);
% [UA, SA, VA] = svd(A, 'econ');
% sA = diag(SA);
% fprintf('cond(A) = %.3e\n', sA(1)/sA(end));
% 
% x = randn(n,1);
% x = x / norm(x);
% z = randn(m,1);
% z = z - UA*(UA'*z);
% z = z - UA*(UA'*z);                     % twice, for orthogonality to working accuracy
% b = A*x + resid * z/norm(z);
% clear UA SA z

sA = diag(SA);
clear SA
fprintf('cond(A) = %.3e, ||r|| = %.1e\n', sA(1)/sA(end), norm(b - A*x));

normAF = norm(A,'fro');
theta  = normAF / norm(b);
fwd = @(xh) norm(xh - x) / norm(x);
bwd = @(xh) kw_estimate(A, b, xh, sA, VA, theta) / normAF;

% S{\&}S rather than S\&S: MATLAB's latex legend escapes a bare \& to "&amp;".
%          name                                              init      refine  color      mark style
methods = { ...
    'Sketch-and-precondition (zero)',                'zero',   false, '#DC267F', 's', '--'; ...
    'Sketch-and-precondition (S{\&}S)',              'sketch', false, '#648FFF', 'o', '-';  ...
    'Sketch-and-precondition (S{\&}S, restarting)',  'sketch', true,  '#009E73', '^', '-.'};
nmeth = size(methods,1);

% Refinement, when used, happens halfway through the budget
refine_point = @(i, k) methods{i,3} * floor(k/2);

%% Left panels: accuracy against iteration

fe = zeros(nmeth, iters+1, 3);
be = zeros(nmeth, iters+1, 3);

wall = tic;
for i = 1:nmeth
    FE = zeros(trials, iters+1);
    BE = zeros(trials, iters+1);
    for t = 1:trials
        [~, X] = SketchAndPrecondition(A, b, iters, methods{i,2}, refine_point(i,iters));
        for j = 1:iters+1
            FE(t,j) = fwd(X(:,j));
            BE(t,j) = bwd(X(:,j));
        end
    end
    fe(i,:,:) = quantile(FE, [0.10 0.50 0.90], 1)';
    be(i,:,:) = quantile(BE, [0.10 0.50 0.90], 1)';
    fprintf('  %-38s forward %.2e  backward %.2e  (median, %d its)\n', ...
            methods{i,1}, fe(i,end,2), be(i,end,2), iters);
end
fprintf('Iteration sweep finished in %.1f s\n', toc(wall));

%% Right panel: accuracy against runtime

% One untimed run of everything first, so JIT and first-call overheads do not
% land on the first timed trial.
SketchAndPrecondition(A, b, 5);
xq = A\b;

% Only the variants without refinement appear on the right panel, so only they
% are timed.  Row a of tm and er belongs to method rmeth(a).
rmeth = find(~[methods{:,3}]);
nr    = numel(rmeth);
nk    = numel(ks);
tm    = zeros(nr, nk, 3);
er    = zeros(nr, nk, 3);

wall = tic;
for a = 1:nr
    i = rmeth(a);
    for j = 1:nk
        tt = zeros(trials,1);
        ee = zeros(trials,1);
        for t = 1:trials
            tic;
            xh = SketchAndPrecondition(A, b, ks(j), methods{i,2}, refine_point(i,ks(j)));
            tt(t) = toc;
            ee(t) = fwd(xh);
        end
        tm(a,j,:) = quantile(tt, [0.10 0.50 0.90]);
        er(a,j,:) = quantile(ee, [0.10 0.50 0.90]);
    end
    fprintf('  %-38s timed\n', methods{i,1});
end

% Baselines.  Sketch-and-solve uses a sketch of size s = 2n, not the 12n the
% preconditioner uses: as a standalone solver it only needs an embedding.
SketchAndPrecondition(A, b, 0, 'sketch', 0, 2*n);       % untimed warm-up
tt = zeros(trials,1);  ee = zeros(trials,1);
for t = 1:trials
    tic;  xh = SketchAndPrecondition(A, b, 0, 'sketch', 0, 2*n);  tt(t) = toc;
    ee(t) = fwd(xh);
end
tm_ss = quantile(tt, [0.10 0.50 0.90]);
er_ss = quantile(ee, [0.10 0.50 0.90]);

tt = zeros(trials,1);
for t = 1:trials
    tic;  xq = A\b;  tt(t) = toc;
end
tm_qr = quantile(tt, [0.10 0.50 0.90]);
fe_qr = fwd(xq);
be_qr = bwd(xq);
fprintf('  sketch-and-solve   forward %.2e   %.3f s\n', er_ss(2), tm_ss(2));
fprintf('  A\\b                forward %.2e   backward %.2e   %.3f s\n', fe_qr, be_qr, tm_qr(2));
fprintf('Runtime sweep finished in %.1f s\n', toc(wall));

%% Plot

% Layout is designed in pixels (1/96 in), but the axes are placed in normalized
% units and the export size is pinned through PaperPosition.  Pixel-unit axes
% overflow whenever the on-screen window is not the requested size -- a docked
% figure, or one the screen cannot fit -- and with PaperPositionMode 'auto' the
% export then clips whatever overflowed.  This way the window can be any size:
% the axes scale with it, and saveas always renders a fig_w-by-fig_h layout.
%
% The two left panels share an iteration axis, so the top one drops its tick
% labels and they sit gap px apart; the right panel spans both, top edge to
% bottom edge, making it twice the height of either left panel plus that gap.
% ax_bottom covers the tick labels and x label (~70 px) plus a two-row legend
% strip beneath them (~100 px from the bottom edge); top_pad leaves room for the top tick label.
fig_w     = 1750;
ax_left   = [150 1005];
ax_width  = 630;
ax_bottom = 200;
ax_h      = 300;           % height of each left panel
gap       = 25;            % between the two left panels
top_pad   = 45;
fig_h     = ax_bottom + 2*ax_h + gap + top_pad;

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h], ...
             'PaperUnits','inches','PaperPositionMode','manual', ...
             'PaperPosition',[0 0 fig_w fig_h]/96,'PaperSize',[fig_w fig_h]/96);
npos = @(l, bot, w, h) [l/fig_w, bot/fig_h, w/fig_w, h/fig_h];

% Report every plot_every iterations, matching the right panel's budgets; the
% final iteration is always included.  Every iterate is still computed above.
its    = unique([0:plot_every:iters, iters]);
xstr   = '\mbox{\boldmath $x$}';
xhat   = '\widehat{\mbox{\boldmath $x$}}';

% Top left: forward error
ax1 = axes('Parent',fig,'Units','normalized', ...
           'Position',npos(ax_left(1), ax_bottom+ax_h+gap, ax_width, ax_h));
hold(ax1,'on')
for i = 1:nmeth
    plot_shaded(its, fe(i,its+1,2), fe(i,its+1,1), fe(i,its+1,3), methods{i,4}, ...
                'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,5}, 'LineStyle', methods{i,6});
end
plot(ax1, [0 iters], [fe_qr fe_qr], 'k--', 'LineWidth', 1.2);
set(ax1,'YScale','log','XTickLabel',[])
ylabel(ax1, ['$\|' xhat '-' xstr '\|/\|' xstr '\|$'], 'Interpreter','latex')
xlim(ax1,[0 iters]); grid(ax1,'on')

% Bottom left: Karlson-Walden backward error estimate
ax2 = axes('Parent',fig,'Units','normalized', ...
           'Position',npos(ax_left(1), ax_bottom, ax_width, ax_h));
hold(ax2,'on')
for i = 1:nmeth
    plot_shaded(its, be(i,its+1,2), be(i,its+1,1), be(i,its+1,3), methods{i,4}, ...
                'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,5}, 'LineStyle', methods{i,6});
end
plot(ax2, [0 iters], [be_qr be_qr], 'k--', 'LineWidth', 1.2);
set(ax2,'YScale','log')
xlabel(ax2,'Iteration','Interpreter','latex')
ylabel(ax2, ['$\widehat{\mathrm{BE}}(' xhat ')/\|\mbox{\boldmath $A$}\|_{\rm F}$'], ...
       'Interpreter','latex')
xlim(ax2,[0 iters]); grid(ax2,'on')
linkaxes([ax1 ax2],'x')

% Right: forward error against runtime
ax3 = axes('Parent',fig,'Units','normalized', ...
           'Position',npos(ax_left(2), ax_bottom, ax_width, 2*ax_h+gap));
hold(ax3,'on')
eb = @(T, E, color, mark) errorbar(ax3, T(:,2), E(:,2), ...
        E(:,2)-E(:,1), E(:,3)-E(:,2), T(:,2)-T(:,1), T(:,3)-T(:,2), ...
        'Color', color, 'Marker', mark, 'LineStyle', 'none', ...
        'MarkerSize', 7, 'LineWidth', 1.2, 'CapSize', 4);
for a = 1:nr
    i = rmeth(a);
    eb(squeeze(tm(a,:,:)), squeeze(er(a,:,:)), methods{i,4}, methods{i,5});
end
eb(tm_ss, er_ss, '#FE6100', 'd');
eb(tm_qr, [fe_qr fe_qr fe_qr], 'k', 'p');
set(ax3,'XScale','log','YScale','log')
% Pad the limits so markers and bars at the extremes (A\b is the slowest point)
% are not pressed against the frame.
tall = [tm(:); tm_ss(:); tm_qr(:)];
eall = [er(:); er_ss(:); fe_qr];
xlim(ax3, [min(tall)/1.3, max(tall)*1.3]);
ylim(ax3, [min(eall)/10, max(eall)*10]);
% The runtimes span less than a decade, so the default log ticks label only one
% point; add intermediate ticks so individual times can be read off.
tk = [0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.5 1 2 3 5 10];
tk = tk(tk >= min(tall)/1.3 & tk <= max(tall)*1.3);
set(ax3, 'XTick', tk, 'XTickLabel', compose('%g', tk));
xlabel(ax3,'Runtime (s)','Interpreter','latex')
ylabel(ax3, ['$\|' xhat '-' xstr '\|/\|' xstr '\|$'], 'Interpreter','latex')
grid(ax3,'on')

% One legend for all three panels.  Its entries are drawn as invisible stand-ins
% so each glyph carries the full style: variants as line plus marker (left-panel
% curves, right-panel markers), sketch-and-solve as a marker, and A\b as a dashed
% line with a marker (dashed reference on the left, point on the right).
h = gobjects(nmeth+2,1);
for i = 1:nmeth
    h(i) = plot(ax3, NaN, NaN, 'Color', methods{i,4}, 'Marker', methods{i,5}, ...
                'LineStyle', methods{i,6}, 'LineWidth', 1.5, 'MarkerSize', 6);
end
h(nmeth+1) = plot(ax3, NaN, NaN, 'Color', '#FE6100', 'Marker', 'd', ...
                  'LineStyle', 'none', 'MarkerSize', 7);
h(nmeth+2) = plot(ax3, NaN, NaN, 'Color', 'k', 'Marker', 'p', ...
                  'LineStyle', '--', 'LineWidth', 1.2, 'MarkerSize', 7);
lgd = legend(h, [methods(:,1); {'Sketch-and-solve'}; ...
                 {'$\mbox{\boldmath $A$}\backslash\mbox{\boldmath $b$}$'}], ...
             'Interpreter','latex', 'NumColumns', 3);
% The legend's size is set by its text, so read it in pixels, where it does not
% depend on the window, then centre it in the designed fig_w-by-fig_h layout.
lgd.Units = 'pixels';
lgd_wh    = lgd.Position(3:4);
lgd.Units = 'normalized';
lgd.Position = npos(max(0, (fig_w - lgd_wh(1))/2), 30, lgd_wh(1), lgd_wh(2));

ansave(fig, fullfile(here,'..','..','figures','sketch_precondition'));

%% Local functions

function be = kw_estimate(A, b, xh, sA, VA, theta)
%KW_ESTIMATE  Karlson-Walden estimate of the least-squares backward error,
%   theta/sqrt(1+theta^2||x||^2) * ||(Sigma^2 + mu I)^{-1/2} V'A'r||,
%   mu = theta^2||r||^2/(1+theta^2||x||^2), from a precomputed SVD of A.
r   = b - A*xh;
nx2 = norm(xh)^2;
mu  = theta^2 * norm(r)^2 / (1 + theta^2*nx2);
g   = VA' * (A'*r);
be  = theta / sqrt(1 + theta^2*nx2) * norm(g ./ sqrt(sA.^2 + mu));
end
