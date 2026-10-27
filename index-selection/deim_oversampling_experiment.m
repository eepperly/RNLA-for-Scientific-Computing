%% Index selection vs amount of oversampling, at fixed noise and basis size
%
% Setup is section 2.3 of Peherstorfer, Drmac and Gugercin, SIAM J. Sci.
% Comput. 42(5):A2837-A2864, 2020: f(x;xi) = exp(-(x-xi)^2/5e-3) on N = 8192
% equidistant points in [-2*pi,2*pi], U the POD basis of 2500 equidistant
% snapshots in xi, reconstruction at one fresh parameter.  Here the noise
% level and basis size are fixed and the sweep is over the number of sampling
% points m.
%
% ACCURACY is measured against the clean field,
%
%     ||F - U*C||_F / ||F||_F,
%
% while the measurements fed to the fit are noisy, C = (D*U(S,:))^+ D*Fn(S,:)
% with Fn = F + noise.  The reference plotted as "optimal" is
%
%     ||F - U*U'*Fn||_F / ||F||_F,
%
% the error of the fit that sees every one of the N noisy entries rather than
% only m of them.  Subsampling cannot beat it, so it is the right floor: it
% already carries the in-range noise that no method can avoid, and the gap
% above it is exactly what subsampling costs.
%
% Two references that do not work here.  Measuring against Fn installs a floor
% of about 1.2e-3, since the noise is full rank over all N dimensions while only
% n lie in range(U), so 98% of its energy is unrepresentable -- larger than the
% amplification being compared, and it hides it.  Using ||F - U*U'*F|| instead
% gives ~3e-15 once n reaches the numerical rank, which is machine zero and
% spans 25 decades on the axis.
%
% The projection residual is formed explicitly rather than as
% ||F||^2 - ||U'F||^2: the basis captures F to machine precision, so that
% subtraction cannot resolve the answer.  It is noise-independent and so is
% computed once.
%
% METHODS.  Three do not oversample and appear as horizontal references:
% CPQR and volume sampling at m = n, and the optimal projection.  Three vary
% with m: leverage and leveraged volume sampling, each using the importance
% weights its distribution supplies, plus GappyPOD+E as a deterministic
% baseline (Algorithm 1 of the paper above, in GPODE).
%
% At m = n GappyPOD+E is exactly CPQR, since its first n points are the QDEIM
% pivots; the two curves meeting at the left edge is a check on both.
%
% The right panel is the point-selection time, which is where these methods
% differ by orders of magnitude: the samplers are essentially free while
% GappyPOD+E does an SVD per added point.

N       = 8192;                 % spatial points
Mtrain  = 2500;                 % snapshot parameters
Mtest   = 1;                    % test parameters
n       = 500;                  % POD modes, fixed
sigma   = 1e-4;                 % noise standard deviation, fixed
facs    = [1 1.2 1.4 1.6 2 2.5 3 4];    % oversampling m/n
trials   = 100;                 % number of trials

ms = round(facs*n);

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'));

datadir = fullfile(here,'..','data');
if ~isfolder(datadir), mkdir(datadir); end
cachefile = fullfile(datadir,'deim_oversampling.mat');
params = struct('N',N,'Mtrain',Mtrain,'Mtest',Mtest,'n',n, ...
                'sigma',sigma,'facs',facs,'trials',trials,'ref','clean-vs-projnoisy');

% Curves split into two groups.  The oversampling rules are solid with markers;
% the m = n references carry no marker and are told apart by colour and line
% style, so they read as baselines rather than as another sampled curve.
%
%          name                      selector                 color      mark    style  oversamples  random
methods = { ...
    'Leverage',           @SelectLeverage,         '#DC267F', 'd',    '-',  true,  true;  ...
    'Leveraged volume',   @SelectLeveragedVolume,  '#FE6100', 'v',    '-',  true,  true;  ...
    'GappyPOD+E',         [],                      '#785EF0', 'o',    '-',  true,  false; ...
    'CPQR',               @SelectCPQR,             '#000000', 'none', '--', false, false; ...
    'Volume sampling',    @SelectVolume,           '#009E73', 'none', '-.', false, true};
nmeth = size(methods,1);
nm    = numel(ms);

rebuild = true;
if isfile(cachefile)
    cached = load(cachefile,'params');
    if isequal(cached.params, params)
        fprintf('Loading results from cache\n');
        load(cachefile,'err','tsel','optv');
        rebuild = false;
    else
        fprintf('Cache built with different parameters; rebuilding.\n');
    end
end

if rebuild

    x = linspace(-2*pi, 2*pi, N)';
    fprintf('Building %d-by-%d snapshots and POD basis...\n', N, Mtrain);
    Ftrain = exp(-(x - linspace(1,3,Mtrain)).^2 / 5e-3);
    [Uall, Sall, ~] = svd(Ftrain, 'econ');
    sv = diag(Sall);
    U  = Uall(:, 1:n);
    clear Ftrain Uall
    fprintf('  numerical rank %d; n = %d, sigma = %.0e\n', ...
            sum(sv > eps*sv(1)), n, sigma);

    rng(0);
    Ftest = exp(-(x - (1 + 2*rand(1,Mtest))).^2 / 5e-3);

    % Deterministic selections, computed once and timed.  GappyPOD+E is greedy
    % and appends one point at a time, so p(1:m) for a smaller m is what it
    % would have returned anyway; it is still called per m so that the reported
    % time is the honest cost of selecting that many points.
    pdet = cell(nmeth, nm);  ddet = cell(nmeth, nm);  tdet = zeros(nmeth, nm);
    for i = 1:nmeth
        if methods{i,7}, continue, end                       % random, do later
        for jm = 1:nm
            m = ms(jm);
            if ~methods{i,6} && jm > 1                       % does not oversample
                pdet{i,jm} = pdet{i,1};  ddet{i,jm} = ddet{i,1};
                tdet(i,jm) = tdet(i,1);  continue
            end
            t0 = tic;
            if isempty(methods{i,2})
                p = gpode(U, m);  d = ones(numel(p),1);      % GappyPOD+E
            else
                [p, d] = methods{i,2}(U, m);
            end
            tdet(i,jm) = toc(t0);
            pdet{i,jm} = p;  ddet{i,jm} = d;
        end
        fprintf('  %-18s selected, %.1f s total\n', methods{i,1}, sum(tdet(i,:)));
    end

    % Noise-independent quantities: the clean projection coefficients and the
    % projection residual, formed directly.
    Gc     = U' * Ftest;
    nF2    = norm(Ftest,'fro')^2;
    proj2  = norm(Ftest - U*Gc, 'fro')^2;

    err  = zeros(nmeth, nm, trials);
    tsel = zeros(nmeth, nm, trials);
    optv = zeros(trials,1);

    wall = tic;
    for rep = 1:trials
        Fn = Ftest + sigma*randn(N, Mtest);          % noisy measurements
        Gn = U' * Fn;                                % fit seeing all N entries
        optv(rep) = sqrt((proj2 + norm(Gn-Gc,'fro')^2)/nF2);

        for i = 1:nmeth
            for jm = 1:nm
                m = ms(jm);
                if ~methods{i,6} && jm > 1           % flat: copy the m = n value
                    err(i,jm,rep)  = err(i,1,rep);
                    tsel(i,jm,rep) = tsel(i,1,rep);
                    continue
                end
                if methods{i,7}
                    t0 = tic;
                    [p, d] = methods{i,2}(U, m);
                    tsel(i,jm,rep) = toc(t0);
                else
                    p = pdet{i,jm};  d = ddet{i,jm};
                    tsel(i,jm,rep) = tdet(i,jm);
                end

                C = lsqminnorm(d .* U(p,:), d .* Fn(p,:));
                err(i,jm,rep) = sqrt((proj2 + norm(Gc-C,'fro')^2)/nF2);
            end
        end
        fprintf('  replicate %2d done, %.1f s elapsed\n', rep, toc(wall));
    end
    fprintf('Sweep finished in %.1f s\n', toc(wall));

    save(cachefile,'err','tsel','optv','params','-v7.3');
end

emid = median(err,3);  elo = min(err,[],3);  ehi = max(err,[],3);
tmid = median(tsel,3); tlo = min(tsel,[],3); thi = max(tsel,[],3);

fprintf('\n%-20s %12s %12s %12s\n','method','err @ 1x','err @ 2x','err @ 4x');
for i = 1:nmeth
    fprintf('%-20s %12.4e %12.4e %12.4e\n', methods{i,1}, ...
            emid(i,1), emid(i,facs==2), emid(i,end));
end
fprintf('%-20s %12.4e\n','optimal', median(optv));

%% Plot

fig_w = 1750; fig_h = 760;
ax_x  = [150 1005];  ax_w = 630;
ax_y  = 235;         ax_h = 400;

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h], ...
             'PaperPositionMode','auto');

ax1 = axes('Parent',fig,'Units','pixels','Position',[ax_x(1) ax_y ax_w ax_h]);
hold(ax1,'on')
for i = 1:nmeth
    plot_shaded(facs, emid(i,:), elo(i,:), ehi(i,:), methods{i,3}, ...
                'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
plot(ax1, facs, median(optv)*ones(size(facs)), ':', 'Color', [0.5 0.5 0.5], ...
     'LineWidth', 1.5);
set(ax1,'YScale','log')
xlabel(ax1,'Oversampling $m/n$','Interpreter','latex')
ylabel(ax1,['$\|\mbox{\boldmath $f$}-\widehat{\mbox{\boldmath $f$}}\|' ...
            '/\|\mbox{\boldmath $f$}\|$'],'Interpreter','latex')
title(ax1,'Accuracy','Interpreter','latex')
xlim(ax1,[min(facs) max(facs)])
% The optimal line sits at ~3e-15: at n = the numerical rank the basis
% captures clean F to machine precision, so the projection error is nil and
% the whole error is noise amplification.  Plotting it unclipped would span
% 25 decades, so the axis is held to the range where the methods differ;
% optimal and leverage at m = n both fall outside it.
ylim(ax1,[2e-4 1e-1])
grid(ax1,'on')
ax1.Toolbar.Visible = 'off';

ax2 = axes('Parent',fig,'Units','pixels','Position',[ax_x(2) ax_y ax_w ax_h]);
hold(ax2,'on')
for i = 1:nmeth
    plot_shaded(facs, tmid(i,:), tlo(i,:), thi(i,:), methods{i,3}, ...
                'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
set(ax2,'YScale','log')
xlabel(ax2,'Oversampling $m/n$','Interpreter','latex')
ylabel(ax2,'Index selection time (s)','Interpreter','latex')
title(ax2,'Runtime','Interpreter','latex')
xlim(ax2,[min(facs) max(facs)]); grid(ax2,'on')
ax2.Toolbar.Visible = 'off';

lines = flipud(findobj(ax1,'Type','Line'));
lgd = legend(lines, [methods(:,1); {'Projection'}], 'Interpreter','latex', ...
             'Units','pixels', 'NumColumns', 4);
lgd.Position(1) = max(0, (fig_w - lgd.Position(3))/2);
lgd.Position(2) = 40;

ansave(fig, fullfile(here,'..','..','figures','deim_oversampling'));
