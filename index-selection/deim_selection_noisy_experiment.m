%% Index selection under noisy measurements, on the PDG (2020) benchmark
%
% Same experimental setup as DEIM_FIGURE2_EXPERIMENT -- section 2.3 of
%
%   B. Peherstorfer, Z. Drmac and S. Gugercin, "Stability of discrete empirical
%   interpolation and gappy proper orthogonal decomposition with randomized and
%   deterministic sampling points", SIAM J. Sci. Comput. 42(5):A2837-A2864, 2020
%
% -- but with their GappyPOD variants replaced by the selection rules in this
% folder.  The function f(x;xi) = exp(-(x-xi)^2/5e-3) is sampled on N = 8192
% equidistant points in [-2*pi,2*pi]; U is the POD basis of 2500 equidistant
% snapshots in xi; the field is reconstructed at 2500 fresh parameters from
% measurements corrupted by Gaussian noise of standard deviation sigma.  The
% error is eq. (8) of that paper: the mean over test parameters of the relative
% error, averaged over 10 noise replicates, with the band showing min and max.
%
% METHODS.  Two at m = n, which interpolate, and two oversampled to m = 2n with
% importance weights, which regress:
%
%   CPQR                       m = n, unweighted.  Deterministic; this is QDEIM.
%   Volume sampling            m = n, unweighted.  Exact VS_n(U) by the fast
%                              rejection sampler REJECTIONRPQR.
%   Leverage, weighted         m = 2n, weights d_i = sqrt(n/(m*ell_i)).
%   Leveraged volume, weighted m = 2n, same weights, volume-sampled core.
%
% The weighted fit solves min ||D*(U(S,:)*c - meas)||, so the DEIM constant
% governing it is ||(D*U(S,:))^+ D||_2.  At m = n that expression collapses to
% ||U(S,:)^{-1}||: the weights cancel against a square U(S,:), which is why
% only the oversampled rules carry them.
%
% Panels (a)-(e) are sigma = 1e-4 down to 1e-8, as in the paper's Figure 2.

N       = 8192;                     % spatial points
Mtrain  = 2500;                     % snapshot parameters
Mtest   = 2500;                     % test parameters
ns      = 100:100:1000;             % DEIM dimensions
sigmas  = 10.^-(4:8);               % noise levels, one panel each
nreps   = 10;                       % noise replicates

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'));

datadir = fullfile(here,'..','data');
if ~isfolder(datadir), mkdir(datadir); end
cachefile = fullfile(datadir,'deim_selection_noisy.mat');
params    = struct('N',N,'Mtrain',Mtrain,'Mtest',Mtest,'ns',ns, ...
                   'sigmas',sigmas,'nreps',nreps);

%          name                                   selector                 factor color      mark  random
methods = { ...
    'CPQR',                                  @SelectCPQR,            1, '#000000', 'o', false; ...
    'Volume sampling',                       @SelectVolume,          1, '#009E73', '^', true;  ...
    'Leverage ($2\times$, weighted)',        @SelectLeverage,        2, '#DC267F', 's', true;  ...
    'Leveraged volume ($2\times$, weighted)',@SelectLeveragedVolume, 2, '#FE6100', 'v', true};
nmeth = size(methods,1);
nn    = numel(ns);
nsig  = numel(sigmas);

rebuild = true;
if isfile(cachefile)
    cached = load(cachefile,'params');
    if isequal(cached.params, params)
        fprintf('Loading results from cache\n');
        load(cachefile,'err','lam');
        rebuild = false;
    else
        fprintf('Cache built with different parameters; rebuilding.\n');
    end
end

if rebuild

    x = linspace(-2*pi, 2*pi, N)';
    fprintf('Building %d-by-%d snapshots and POD basis...\n', N, Mtrain);
    tb = tic;
    Ftrain = exp(-(x - linspace(1,3,Mtrain)).^2 / 5e-3);
    [Uall, Sall, ~] = svd(Ftrain, 'econ');
    sv   = diag(Sall);
    Uall = Uall(:, 1:max(ns));
    clear Ftrain
    fprintf('  %.1f s; numerical rank %d, sigma_1 = %.2e\n', ...
            toc(tb), sum(sv > eps*sv(1)), sv(1));

    rng(0);
    Ftest = exp(-(x - (1 + 2*rand(1,Mtest))).^2 / 5e-3);
    fnorm = vecnorm(Ftest);

    err = zeros(nmeth, nn, nsig, nreps);
    lam = zeros(nmeth, nn, nreps);

    wall = tic;
    for jn = 1:nn
        n = ns(jn);
        U = Uall(:, 1:n);
        G  = U' * Ftest;
        Rn = vecnorm(Ftest - U*G);          % projection residual, once per n

        % Deterministic rules are selected once and reused across replicates.
        pfix = cell(nmeth,1);  dfix = cell(nmeth,1);
        for i = 1:nmeth
            if ~methods{i,6}
                [pfix{i}, dfix{i}] = methods{i,2}(U, methods{i,3}*n);
            end
        end

        for rep = 1:nreps
            for i = 1:nmeth
                if methods{i,6}
                    [p, d] = methods{i,2}(U, methods{i,3}*n);
                else
                    p = pfix{i};  d = dfix{i};
                end

                Us = d .* U(p,:);                       % D*U(S,:)
                lam(i,jn,rep) = norm(pinv(Us) .* d');   % ||(D*U(S,:))^+ D||_2

                % The solve is linear in the noise, so factor once against both
                % right-hand sides and rebuild C = C0 + sigma*Cz per panel.
                Zn   = randn(numel(p), Mtest);
                Cb   = lsqminnorm(Us, d .* [Ftest(p,:), Zn]);
                C0   = Cb(:, 1:Mtest);
                Cz   = Cb(:, Mtest+1:end);

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

    save(cachefile,'err','lam','params','-v7.3');
end

emid = mean(err, 4);  elo = min(err, [], 4);  ehi = max(err, [], 4);

fprintf('\nDEIM constant at n = %d (median over replicates):\n', ns(end));
for i = 1:nmeth
    fprintf('  %-42s %.3e\n', methods{i,1}, median(lam(i,end,:)));
end

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
                    methods{i,4}, 'LineWidth', 1.5, 'MarkerSize', 5, ...
                    'Marker', methods{i,5});
    end
    set(ax,'YScale','log')
    xlabel(ax,'DEIM dimension $n$','Interpreter','latex')
    ylabel(ax,'rel. state error','Interpreter','latex')
    title(ax, sprintf('(%s) $\\sigma = 10^{-%d}$', panel(js), js+3), ...
          'Interpreter','latex')
    xlim(ax,[min(ns) max(ns)]); grid(ax,'on')
    ax.Toolbar.Visible = 'off';
    if js == nsig
        lines = flipud(findobj(ax,'Type','Line'));
        lgd = legend(lines, methods(:,1), 'Interpreter','latex', ...
                     'Units','pixels', 'NumColumns', 1);
        lgd.Position(1) = ax_x(3);
        lgd.Position(2) = ax_y(2) + ax_h/2;
    end
end

ansave(fig, fullfile(here,'..','..','figures','deim_selection_noisy'));
