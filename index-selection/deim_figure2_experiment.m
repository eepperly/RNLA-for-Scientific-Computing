%% Reproduction of Figure 2 of Peherstorfer, Drmac and Gugercin (2020)
%
%   B. Peherstorfer, Z. Drmac and S. Gugercin, "Stability of discrete empirical
%   interpolation and gappy proper orthogonal decomposition with randomized and
%   deterministic sampling points", SIAM J. Sci. Comput. 42(5):A2837-A2864,
%   2020.
%
% Setup is section 2.3 of that paper.  The parametrized function
%
%     f(x;xi) = exp( -(x - xi)^2 / (5e-3) ),      eq. (7)
%
% is sampled on N = 8192 equidistant points in [-2*pi, 2*pi], with parameter
% xi in [1,3].  Snapshots are taken at 2500 equidistant xi and U is their POD
% basis; the function is then reconstructed at 2500 further parameters drawn
% uniformly in [1,3], from measurements corrupted by iid Gaussian noise of
% standard deviation sigma, eq. (9).  The reported error is the averaged
% relative state error of eq. (8), averaged again over 10 noise replicates,
% with the band showing the minimum and maximum over replicates.
%
% Section 6.1 compares, at m = 2n sampling points:
%
%   QDEIM        m = n points from column-pivoted QR.  The unstable baseline:
%                the fit interpolates, so it reproduces the noise exactly.
%   GappyPOD+E   the paper's Algorithm 1, in GPODE.
%   GappyPOD+R   the n QDEIM points, then m-n uniformly at random with
%                replacement.
%   GappyPOD+L   the n QDEIM points, then m-n by leverage scores.
%
% NOT REPRODUCED: GappyPOD+D, which takes its extra points by the greedy
% extension of [Carlberg, Farhat, Cortial and Amsallem, JCP 2013, Algorithm 4].
% That paper was not available here, so the curve is omitted rather than
% guessed at.
%
% Panels (a)-(e) are sigma = 1e-4 down to 1e-8, as in the paper.
%
% NOTE ON THE BASIS.  The snapshot matrix has numerical rank about 105: the
% singular values reach 1e-14 by n = 200 and are flat at 6.7e-15 thereafter.
% So for most of the sweep U's trailing columns are roundoff, not signal.  That
% is not a defect of the reproduction -- it is the mechanism.  The noiseless
% error saturates near 1e-14 once the basis spans the data, while
% ||(P'U)^{-1}|| climbs from 5.5 at n = 100 to 83 at n = 1000, so each extra
% (null) mode buys nothing and amplifies the noise a little more.

N       = 8192;                     % spatial points
Mtrain  = 2500;                     % snapshot parameters
Mtest   = 2500;                     % test parameters
ns      = 100:100:1000;             % DEIM dimensions
sigmas  = 10.^-(4:8);               % noise levels, one panel each
nreps   = 10;                       % noise replicates

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'));

%% Snapshots, POD basis, test fields

x = linspace(-2*pi, 2*pi, N)';

fprintf('Building %d-by-%d snapshots and POD basis...\n', N, Mtrain);
tb = tic;
Ftrain = exp(-(x - linspace(1,3,Mtrain)).^2 / 5e-3);
[Uall, Sall, ~] = svd(Ftrain, 'econ');
sv   = diag(Sall);
Uall = Uall(:, 1:max(ns));
clear Ftrain
fprintf('  %.1f s; sigma_1 = %.2e, sigma_%d = %.2e (ratio %.1e)\n', ...
        toc(tb), sv(1), max(ns), sv(max(ns)), sv(max(ns))/sv(1));

rng(0);
Ftest = exp(-(x - (1 + 2*rand(1,Mtest))).^2 / 5e-3);
fnorm = vecnorm(Ftest);

%          name           color      marker
methods = { ...
    'QDEIM',        '#000000', 'o'; ...
    'GappyPOD+E',   '#DC267F', 's'; ...
    'GappyPOD+R',   '#648FFF', 'd'; ...
    'GappyPOD+L',   '#FE6100', '^'};
nmeth = size(methods,1);
nn    = numel(ns);
nsig  = numel(sigmas);

err = zeros(nmeth, nn, nsig, nreps);

% The sweep costs ~10 minutes, almost all of it GPODE at large n, while the
% results are a few thousand doubles.  Cache them so the figure can be
% retouched without recomputing.
datadir = fullfile(here,'..','data');
if ~isfolder(datadir), mkdir(datadir); end
cachefile = fullfile(datadir,'deim_pdg_figure2.mat');
params    = struct('N',N,'Mtrain',Mtrain,'Mtest',Mtest, ...
                   'ns',ns,'sigmas',sigmas,'nreps',nreps);

if isfile(cachefile)
    cached = load(cachefile,'params');
    if isequal(cached.params, params)
        fprintf('Loading sweep results from cache\n');
        load(cachefile,'err');
        params.done = true;
    end
end

%% Sweep

wall = tic;
if ~isfield(params,'done')
    for jn = 1:nn
        n = ns(jn);
        m = 2*n;
        U = Uall(:, 1:n);

        % Per-n quantities.  The reconstruction error splits orthogonally as
        % f - U*c = (I-U*U')f + U*(U'f - c), so the column norms of the residual
        % need only be formed once.
        G   = U' * Ftest;
        Rn  = vecnorm(Ftest - U*G);
        ell = sum(U.^2, 2);

        [~, ~, pq] = qr(U', 'vector');
        pqdeim = pq(1:n)';                          % QDEIM points, m = n
        pE     = gpode(U, m);                       % GappyPOD+E, deterministic

        for rep = 1:nreps
            pR = [pqdeim; randi(N, m-n, 1)];                    % uniform, with replacement
            pL = [pqdeim; randsample(N, m-n, true, ell)];       % leverage, with replacement
            psets = {pqdeim, pE, pR, pL};

            for i = 1:nmeth
                p  = psets{i};
                Up = U(p,:);

                % The solve is linear in the noise, so factor the two pieces once
                % and rebuild C = C0 + sigma*Cz for every noise level.  Each panel
                % is a valid realization; they share the underlying draw.
                C0 = Up \ Ftest(p,:);
                Cz = Up \ randn(numel(p), Mtest);

                for js = 1:nsig
                    C = C0 + sigmas(js)*Cz;
                    e = sqrt(Rn.^2 + vecnorm(G - C).^2) ./ fnorm;   % eq. (8)
                    err(i, jn, js, rep) = mean(e);
                end
            end
        end
        fprintf('  n = %4d done, %.1f s elapsed\n', n, toc(wall));
    end
fprintf('Sweep finished in %.1f s\n', toc(wall));
save(cachefile,'err','params');
end

emid = mean(err, 4);
elo  = min(err, [], 4);
ehi  = max(err, [], 4);

%% Plot: one panel per noise level

fig_w = 1750; fig_h = 980;
ax_w  = 430;  ax_h = 300;
ax_x  = [130 700 1270];
ax_y  = [600 195];

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h], ...
             'PaperPositionMode','auto');

panel = 'abcde';
for js = 1:nsig
    r  = ceil(js/3);
    c  = js - 3*(r-1);
    ax = axes('Parent',fig,'Units','pixels', ...
              'Position',[ax_x(c) ax_y(r) ax_w ax_h]);
    hold(ax,'on')
    for i = 1:nmeth
        plot_shaded(ns, emid(i,:,js), elo(i,:,js), ehi(i,:,js), ...
                    methods{i,2}, 'LineWidth', 1.5, 'MarkerSize', 5, ...
                    'Marker', methods{i,3});
    end
    set(ax,'YScale','log')
    xlabel(ax,'DEIM dimension $n$','Interpreter','latex')
    ylabel(ax,'rel. state error','Interpreter','latex')
    title(ax, sprintf('(%s) $\\sigma = 10^{-%d}$', panel(js), js+3), ...
          'Interpreter','latex')
    xlim(ax,[min(ns) max(ns)])
    ylim(ax, sigmas(js)*[1.5 40])   % headroom: QDEIM otherwise sits on the frame
    grid(ax,'on')
    ax.Toolbar.Visible = 'off';
    if js == nsig
        lines = flipud(findobj(ax,'Type','Line'));
        lgd = legend(lines, methods(:,1), 'Interpreter','latex', ...
                     'Units','pixels', 'NumColumns', 1);
        lgd.Position(1) = ax_x(3);
        lgd.Position(2) = ax_y(2) + ax_h/2;
    end
end

ansave(fig, fullfile(here,'..','..','figures','deim_pdg_figure2'));
