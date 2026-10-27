%% Trace estimation of the TFIM partition function
%
% Estimate the partition function Z(beta) = tr(exp(-beta*H)) of the
% transverse-field Ising Hamiltonian H on an open chain of L spins at the
% critical point J = h = 1, so n = 2^L.  The matrix A = f(H) is never formed:
% every matvec X -> A*X is computed by a fixed number of Lanczos steps
% (LANCZOS_FUN, nlanczos = 60), the same for every algorithm.  One matvec with
% A therefore costs exactly nlanczos matvecs with H, so the x-axis is
% proportional to the true cost.
%
% With the step count fixed, the matvecs are approximate, and nothing refines
% them.  The errors plotted are against the exact partition function, so they
% contain the Lanczos approximation error as well as each estimator's sampling
% error.  Where the sampling error falls below what 60 steps can resolve --
% most likely at large beta, where exp(-beta*H) is hardest to approximate by
% a low-degree polynomial -- the curves should level off at a common floor
% set by Lanczos rather than by the estimator.
%
% A is normalized as A = exp(-beta*(H - lam_min*I)), so ||A|| = 1 and
% Z = exp(-beta*lam_min) * tr(A).  The normalization only rescales A by a
% known constant -- relative errors are unchanged -- and it keeps exp() from
% overflowing at large beta.  The exact Z comes from the closed-form
% free-fermion spectrum (TFIM_EIGENVALUES).
%
% Five estimators, all with resphering (see RESPHERE):
%
%   Girard-Hutchinson  spherical probes (resphering with no low-rank part)
%   Hutch++            probes resphered onto range(Q)^perp, Q = orth(A*Omega)
%   Nystrom++          probes resphered onto range(Omega)^perp; single pass
%   XTrace             each left-out vector resphered onto range(Q_(i))^perp
%   XNysTrace          each left-out vector resphered onto range(Omega_{-i})^perp
%
% Three inverse temperatures set the spectral decay of A:
%
%   beta = 0.02  little decay    all eigenvalues within a factor of ~2;
%                                tr(A) = 0.75 n
%   beta = 1     moderate decay  tr(A) ~ 5.8; 196 eigenvalues above 1e-3
%   beta = 3     heavy decay     ground state carries 60% of the trace;
%                                11 eigenvalues above 1e-3
%
% The regimes are calibrated by how the estimators behave, not just by the
% spread of the spectrum.  beta = 0.3 already spans four decades, but its bulk
% of ~100+ comparable eigenvalues is too fat for a rank <= 96 approximation to
% remove much variance: every method there converges at the Monte Carlo rate
% and they tie within noise, indistinguishable from little decay.  At beta = 1
% Girard-Hutchinson still converges as M^{-1/2} while the variance-reduced
% methods run at about M^{-1.35} -- clearly faster, but not yet the spectral
% convergence (about M^{-4} here) of beta = 3.
%
% Each panel plots relative error against the number of matvecs with A.

L      = 12;                        % spins; n = 2^L
betas  = [0.02 1 3];
regime = {'Little decay', 'Moderate decay', 'Heavy decay'};
Ms     = [10 20 40 80 160];         % matvec budgets
trials = 1000;                      % repetitions, for the median and 10/90 bands
nlanczos = 60;                      % Lanczos steps per matvec with A, fixed

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'), fullfile(here,'..','sde'));

H    = tfim(L);
n    = size(H,1);
lam  = tfim_eigenvalues(L);         % exact spectrum, closed form
lmin = lam(1);

%          name                  estimator            colour     mark style
methods = { ...
    'Girard--Hutchinson', @girard_hutchinson, '#000000', 'o', '-';  ...
    'Hutch++',            @hutchpp,           '#648FFF', 's', '--'; ...
    'Nystr\"om++',        @nystrompp,         '#785EF0', 'd', ':';  ...
    'XTrace',             @xtrace,            '#DC267F', '^', '-';  ...
    'XNysTrace',          @xnystrace,         '#FE6100', 'v', '-.'};
nmeth = size(methods,1);
nM    = numel(Ms);
nb    = numel(betas);

% [method, budget, quantile (10/50/90), beta]
err = zeros(nmeth, nM, 3, nb);
mvs = zeros(nmeth, nM, nb);         % matvecs actually used

%% Sweep

wall = tic;
for q = 1:nb
    beta = betas(q);
    f    = @(x) exp(-beta*(x - lmin));
    Afun = @(X) lanczos_fun(H, f, X, nlanczos);
    trA  = sum(f(lam));

    fprintf('beta = %.2f (%s): tr(A) = %.6e, Z = %.10e\n', ...
            beta, regime{q}, trA, exp(-beta*lmin)*trA);

    for i = 1:nmeth
        for j = 1:nM
            e = zeros(trials,1);
            for t = 1:trials
                [est, mvs(i,j,q)] = methods{i,2}(Afun, n, Ms(j));
                e(t) = abs(est - trA) / trA;
            end
            err(i,j,:,q) = quantile(e, [0.10 0.50 0.90]);
        end
        fprintf('  %-20s done, %6.1f s elapsed\n', methods{i,1}, toc(wall));
    end
end
fprintf('Sweep finished in %.1f s\n', toc(wall));

%% Plot: one row of three panels, one per spectral-decay regime

% Pixel geometry, so exports have a fixed size regardless of the screen.
fig_w     = 1750;
fig_h     = 640;
ax_left   = [140 705 1270];
ax_width  = 440;
ax_bottom = 215;
ax_height = 340;

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h], ...
             'PaperPositionMode','auto');

for q = 1:nb
    ax = axes('Parent',fig,'Units','pixels', ...
              'Position',[ax_left(q) ax_bottom ax_width ax_height]);
    hold(ax,'on')
    for i = 1:nmeth
        plot_shaded(mvs(i,:,q), err(i,:,2,q), err(i,:,1,q), err(i,:,3,q), ...
                    methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                    'Marker', methods{i,4}, 'LineStyle', methods{i,5});
    end
    set(ax,'XScale','log','YScale','log')
    axis([-Inf Inf 1e-10 1e0])
    xlim(ax,[min(Ms) max(Ms)]); grid(ax,'on')
    xlabel(ax,'Matrix--vector products with $\mbox{\boldmath $A$}$','Interpreter','latex')
    if q == 1
        ylabel(ax,'Relative error','Interpreter','latex')
    end
    title(ax, sprintf('%s, $\\beta = %g$', regime{q}, betas(q)), 'Interpreter','latex')
end

lines = flipud(findobj(ax,'Type','Line'));
lgd = legend(lines, methods(:,1), 'Interpreter','latex', ...
             'NumColumns', nmeth, 'Units','pixels');
lgd.Position(1) = max(0, (fig_w - lgd.Position(3))/2);
lgd.Position(2) = 40;

ansave(fig, fullfile(here,'..','..','figures','trace_tfim'));
