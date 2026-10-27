%% DEIM index selection: comparing five selection rules
%
% The discrete empirical interpolation method approximates a vector f from its
% entries on a small index set S, given a basis U with orthonormal columns:
%
%     f ~ U * (U(S,:) \ f(S)).
%
% With oversampling the fit is a weighted least-squares solve,
%
%     f ~ U * argmin_c ||D*(U(S,:)*c - f(S))||,
%
% where D holds the importance weights of the sampling distribution, and the
% error obeys
%
%     ||f - DEIM(f)|| <= Lambda * ||(I - U*U')f||,
%     Lambda = ||(D*U(S,:))^+ D||_2.
%
% The projection error is fixed by the basis, so the only thing a selection
% rule controls is the amplification constant Lambda.  That is the left panel;
% the right panel is the error the method actually commits on the snapshots.
%
% Two facts make the picture simpler than it looks.  At s = k the matrix U(S,:)
% is square, so (D*U(S,:))^+ D = U(S,:)^{-1} D^{-1} D = U(S,:)^{-1}: the weights
% cancel and Lambda does not depend on D at all.  And uniform sampling has
% constant weights, so D is a multiple of the identity and cancels for it at
% every s.  Weighting therefore only does anything for the leverage-based rules
% under oversampling.
%
% TEST PROBLEM.  The benchmark of section 2.3 of
%
%   B. Peherstorfer, Z. Drmac and S. Gugercin, "Stability of discrete empirical
%   interpolation and gappy proper orthogonal decomposition with randomized and
%   deterministic sampling points", SIAM J. Sci. Comput. 42(5):A2837-A2864,
%   2020,
%
% namely the parametrized function
%
%     f(x;xi) = exp( -(x - xi)^2 / 5e-3 ),   x in [-2*pi, 2*pi],  xi in [1,3],
%
% a narrow Gaussian pulse that translates with the parameter.  Snapshots are
% taken at equispaced xi, the basis U is their POD basis, and the field is then
% reconstructed at fresh test parameters from samples at the selected indices.
%
% WHY THIS ONE.  The measurements are noisy, and that is what makes
% oversampling necessary rather than merely helpful.  With s = k the fit is an
% interpolation: it matches the sampled values exactly, noise included, and the
% paper shows the resulting error grows like sqrt(k) with the reduced dimension
% -- more modes make the reconstruction worse, not better.  With s > k the fit
% becomes a regression over more samples than unknowns, which averages the
% noise down.  On a noiseless problem oversampling buys almost nothing, which
% is why a noiseless test problem could not show this.
%
% A translating narrow pulse is also the hard case for POD: the snapshots are
% nearly shifted copies, the singular values decay slowly, and the modes are
% localized, so the leverage scores are far from flat.
%
% METHODS.  Column-pivoted QR (deterministic, the strong baseline) and uniform
% sampling (the weak one) bracket three structure-aware randomized rules:
% leverage scores, volume sampling by fast randomly pivoted QR, and leveraged
% volume sampling.  Volume sampling and leveraged volume sampling coincide in
% distribution at s = k, so their solid curves agreeing there is a check on both
% implementations rather than a redundancy.
%
% Solid lines select s = k rows, dashed lines oversample to s = 2k.  Volume
% sampling appears solid only: it cannot draw more than k rows of a rank-k
% matrix.  Column-pivoted QR is shown solid as the deterministic reference.

% Mtrain has to be large enough that the POD basis still has numerical rank at
% max(ks): with only 400 snapshots the singular values reach 1e-13 by k = 120,
% the basis is numerically exhausted, and the sweep cannot run far enough for
% the sqrt(k) noise growth to appear.
N      = 200000;           % spatial grid points
Mtrain = 1000;             % snapshot parameters
Mtest  = 400;              % test parameters
sigma  = 1e-4;             % noise standard deviation on the measurements
ks     = 50:50:500;        % basis dimensions to sweep
oversample = 2;            % s = oversample*k on the dashed curves
trials = 20;               % repetitions for the randomized rules

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'));

%% Snapshots, POD basis, and test fields

x = linspace(-2*pi, 2*pi, N)';

xitrain = linspace(1, 3, Mtrain);
fprintf('Building %d-by-%d snapshots...\n', N, Mtrain);
Ftrain = exp(-(x - xitrain).^2 / 5e-3);

fprintf('POD basis by economy SVD...\n');
ts = tic;
[Uall, Sall, ~] = svd(Ftrain, 'econ');
sv   = diag(Sall);
Uall = Uall(:, 1:max(ks));              % only the leading modes are needed
clear Ftrain                            % free before building the test fields
fprintf('  %.1f s\n', toc(ts));

rng(0);
xitest = 1 + 2*rand(1, Mtest);          % fresh parameters, not the training set
Ftest  = exp(-(x - xitest).^2 / 5e-3);
normF  = norm(Ftest, 'fro');

fprintf('N = %d unknowns, %d snapshots; sigma_1 = %.2e, sigma_%d = %.2e\n', ...
        N, Mtrain, sv(1), max(ks)+1, sv(max(ks)+1));
ellmax = sum(Uall.^2, 2);
fprintf('coherence N*max(ell)/k = %.1f (1 = perfectly flat)\n', ...
        N*max(ellmax)/max(ks));
fprintf('measurement noise sigma = %.1e\n', sigma);

%          name                 selector                 color      mark  random  s/k
methods = { ...
    'Column-pivoted QR', @SelectCPQR,            '#000000', 'o', false, 1;   ...
    'Volume sampling',   @SelectVolume,          '#009E73', '^', true,  1;   ...
    'Uniform',           @SelectUniform,         '#648FFF', 's', true,  [1 oversample]; ...
    'Leverage scores',   @SelectLeverage,        '#DC267F', 'd', true,  [1 oversample]; ...
    'Leveraged volume',  @SelectLeveragedVolume, '#FE6100', 'v', true,  [1 oversample]};

% Flatten to one entry per curve.  Colour and marker identify the method,
% line style the oversampling: solid for s = k, dashed for s = oversample*k.
curves = cell(0,7);
for i = 1:size(methods,1)
    for c = methods{i,6}
        if c == 1
            nm = methods{i,1};  ls = '-';
        else
            nm = sprintf('%s ($%d\\times$)', methods{i,1}, c);  ls = '--';
        end
        curves(end+1,:) = {nm, methods{i,2}, methods{i,3}, ...
                           methods{i,4}, ls, methods{i,5}, c};   %#ok<SAGROW>
    end
end
nmeth = size(curves,1);
nk    = numel(ks);

lam = zeros(nmeth, nk, 3);      % DEIM constant, 10/50/90 quantiles
rec = zeros(nmeth, nk, 3);      % relative reconstruction error

% Per-k quantities that do not depend on the selection rule.  Forming U*C
% explicitly for every trial would cost O(N*k*Mtest) each; instead note that
% Ftest - U*C = (I-U*U')Ftest + U*(U'Ftest - C) splits into orthogonal pieces,
% so
%
%     ||Ftest - U*C||_F^2 = ||(I-U*U')Ftest||_F^2 + ||U'Ftest - C||_F^2,
%
% and the second term is only k-by-Mtest.  Both terms are nonnegative, so this
% stays accurate where expanding the square would cancel.
% The columns of Uall are nested, so U(:,1:k)'*Ftest is the first k rows of
% Uall'*Ftest; one product gives every k.
Gfull = Uall' * Ftest;
Gk    = cell(nk,1);
projk = zeros(nk,1);
for j = 1:nk
    Gk{j}    = Gfull(1:ks(j), :);                           % U'Ftest
    projk(j) = max(0, normF^2 - norm(Gk{j},'fro')^2);       % ||(I-U*U')Ftest||^2
end

% Noiseless floor: what the reconstruction would achieve with exact samples.
projerr = sqrt(projk) / normF;

wall = tic;
for i = 1:nmeth
    sel  = curves{i,2};
    fac  = curves{i,7};
    nrep = trials;
    if ~curves{i,6}
        nrep = 1;                       % deterministic, one run suffices
    end

    for j = 1:nk
        k = ks(j);
        U = Uall(:,1:k);
        s = fac * k;
        G = Gk{j};

        tl = zeros(nrep,1);
        tr = zeros(nrep,1);
        for t = 1:nrep
            [S, d] = sel(U, s);

            Us    = d .* U(S,:);                % D*U(S,:)
            tl(t) = norm(pinv(Us) .* d');       % ||(D*U(S,:))^+ D||_2

            % Noisy measurements at the selected indices.  With s = k this is
            % an interpolation and the noise is reproduced exactly; with s > k
            % it is a regression over more samples than unknowns.
            meas = Ftest(S,:) + sigma*randn(s, Mtest);

            % lsqminnorm, not backslash: a bad index set can leave U(S,:)
            % numerically singular, and the minimum-norm solve reports that as
            % a large finite error instead of a warning and a NaN.
            C     = lsqminnorm(Us, d .* meas);
            tr(t) = sqrt(projk(j) + norm(G - C,'fro')^2) / normF;
        end
        lam(i,j,:) = quantile(tl, [0.10 0.50 0.90]);
        rec(i,j,:) = quantile(tr, [0.10 0.50 0.90]);
    end
    fprintf('  %-24s done, %.1f s elapsed\n', curves{i,1}, toc(wall));
end
fprintf('Sweep finished in %.1f s\n', toc(wall));

%% Plot

% Pixel geometry, so the exported width is deterministic and the figure drops
% into LaTeX at a predictable scale.
fig_w     = 1750;
fig_h     = 710;
ax_left   = [150 1005];
ax_width  = 630;
ax_bottom = 235;       % room below for the x label and the legend strip
ax_height = 375;

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h], ...
             'PaperPositionMode','auto');

ax1 = axes('Parent',fig,'Units','pixels', ...
           'Position',[ax_left(1) ax_bottom ax_width ax_height]);
hold(ax1,'on')
for i = 1:nmeth
    plot_shaded(ks, lam(i,:,2), lam(i,:,1), lam(i,:,3), ...
                curves{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', curves{i,4}, 'LineStyle', curves{i,5});
end
set(ax1,'YScale','log')
xlabel(ax1,'Basis dimension $k$','Interpreter','latex')
ylabel(ax1,['$\Lambda = \|(\mbox{\boldmath $D$}\mbox{\boldmath $U$}(S,:))^\dagger' ...
            '\mbox{\boldmath $D$}\|_2$'],'Interpreter','latex')
title(ax1,'DEIM constant','Interpreter','latex')
ylim(ax1,[1 1e8])       % uniform runs far above this; clipped to stay readable
grid(ax1,'on')
ax1.Toolbar.Visible = 'off';        % keep it out of the exported image

ax2 = axes('Parent',fig,'Units','pixels', ...
           'Position',[ax_left(2) ax_bottom ax_width ax_height]);
hold(ax2,'on')
for i = 1:nmeth
    plot_shaded(ks, rec(i,:,2), rec(i,:,1), rec(i,:,3), ...
                curves{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', curves{i,4}, 'LineStyle', curves{i,5});
end
plot(ax2, ks, projerr, 'k:', 'LineWidth', 1.5);      % noiseless floor
set(ax2,'YScale','log')
xlabel(ax2,'Basis dimension $k$','Interpreter','latex')
ylabel(ax2,['$\|\mbox{\boldmath $F$}-\widehat{\mbox{\boldmath $F$}}\|_{\rm F}' ...
            '/\|\mbox{\boldmath $F$}\|_{\rm F}$'],'Interpreter','latex')
title(ax2,'Reconstruction error, noisy samples','Interpreter','latex')
ylim(ax2,[1e-6 1e2])    % likewise; the failures leave the top of the frame
grid(ax2,'on')
ax2.Toolbar.Visible = 'off';

lines = flipud(findobj(ax2,'Type','Line'));
lgd = legend(lines, [curves(:,1); {'Noiseless floor'}], 'Interpreter','latex', ...
             'NumColumns', 5, 'Units','pixels');
lgd.Position(1) = max(0, (fig_w - lgd.Position(3))/2);
lgd.Position(2) = 40;

ansave(fig, fullfile(here,'..','..','figures','deim_selection'));
