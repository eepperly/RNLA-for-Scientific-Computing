%% Column Nystrom approximation: cost and a downstream learning task
%
% Compares four ways of choosing landmarks S for a column Nystrom
% approximation A<S> = A(:,S) A(S,S)^+ A(S,:) of a kernel matrix A:
%
%   Uniform                  S uniform without replacement      UNIFORM_NYSTROM
%   RLS                      recursive ridge leverage scores    RLS_NYSTROM
%   Greedy                   greedy pivoted Cholesky            GREEDY_CHOLESKY
%   Accelerated RPCholesky   block rejection-sampled RPCholesky ACC_RPCHOLESKY
%
% recording, for each run:
%
%   trace error      (tr A - tr A<S>) / tr A, the relative trace-norm error of
%                    the approximation (NYSTROM_TRACE_ERROR)
%   test error       misclassification of kernel ridge regression restricted to
%                    the landmarks (NYSTROM_KRR), on held-out data
%   runtime          wall-clock time to produce S and the columns A(:,S)
%   entry accesses   kernel entries evaluated for the same work, counted
%                    exactly by KERNELORACLE (tabulated and saved, not plotted)
%
% The figure is a 2-by-2 grid: trace error on top, test error below; against
% the number of landmarks k on the left and against runtime on the right.
%
% PROTOCOL.  Every choice below was fixed before any comparison was run, by a
% standard rule that does not involve the four methods, and none was revisited
% afterward:
%
%   Data         UCI Adult, as shipped with MATLAB (census1994.mat): 32561
%                training and 16281 test points, label +1 for income >50K and -1
%                otherwise.  Numeric features z-scored with training statistics;
%                categorical features one-hot encoded over the training
%                categories, with missing values as their own category.
%   Kernel       Gaussian, bandwidth by the median heuristic: the median
%                pairwise distance among 2000 random training points.
%   lambda       chosen once, by exact KRR fit on 5000 random training points and
%                validated on 5000 others, minimizing validation MSE over
%                10.^(-8:0).  Exact KRR, so the choice involves none of the
%                methods being compared.
%   Methods      each at the default of its reference implementation.
%   Ranks        k = 50, 100, 200, 400, 800, doubling.
%   Trials       10 per (method, k); medians with 10/90 quantile bands.
%   Order        the four methods run in a fresh random order in every trial, so
%                no method is systematically timed first or last.
%   Timing       one untimed warm-up call per method before the sweep.
%
% Runtime counts selection plus every kernel evaluation needed to deliver the
% columns A(:,S).  It excludes the KRR solve, the test predictions and the
% trace-error evaluation, which for a given k are the same computation
% whichever method chose S.  Note what that implies: greedy and accelerated
% RPCholesky build the Nystrom factor F as part of selection, so its cost is
% inside their runtime, while uniform and RLS stop at S and A(:,S).

rng('default');

ks     = [50 100 200 400 800];
trials = 10;

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'));

%% Data

D = load('census1994.mat');
[Xtr, ytr, Xte, yte] = encode_adult(D.adultdata, D.adulttest);
ntr = size(Xtr,1);
fprintf('Adult: %d train, %d test, %d features; fraction positive %.3f train, %.3f test\n', ...
        ntr, size(Xte,1), size(Xtr,2), mean(ytr > 0), mean(yte > 0));

%% Bandwidth: median heuristic

idx   = randsample(ntr, 2000);
sq    = sum(Xtr(idx,:).^2, 2);
D2    = max(sq + sq' - 2*(Xtr(idx,:)*Xtr(idx,:)'), 0);
dist  = sqrt(D2(triu(true(2000), 1)));
sigma = median(dist);
fprintf('Bandwidth (median heuristic): sigma = %.4f\n', sigma);

%% lambda: exact KRR on a training subsample, validated on another

perm  = randperm(ntr);
fit   = perm(1:5000);
val   = perm(5001:10000);
Afit  = gauss_kernel(Xtr(fit,:), Xtr(fit,:), sigma);
Aval  = gauss_kernel(Xtr(val,:), Xtr(fit,:), sigma);
[V, E] = eig((Afit + Afit')/2);
e     = diag(E);
Vy    = V' * ytr(fit);
grid_lam = 10.^(-8:0);
valmse   = zeros(size(grid_lam));
for g = 1:numel(grid_lam)
    alpha     = V * (Vy ./ (e + grid_lam(g) * numel(fit)));
    valmse(g) = mean((Aval*alpha - ytr(val)).^2);
end
[~, gbest] = min(valmse);
lam = grid_lam(gbest);
fprintf('lambda by exact-KRR validation: %g   (validation MSE over grid: %s)\n', ...
        lam, mat2str(valmse, 4));
clear Afit Aval V E

%% Sweep

%          name                      selector            colour     mark style
methods = { ...
    'Uniform',                 @uniform_nystrom,  '#648FFF', 'o', '-';  ...
    'RLS',                     @rls_nystrom,      '#785EF0', 's', '--'; ...
    'Greedy',                  @greedy_cholesky,  '#FE6100', 'd', ':';  ...
    'Accelerated RPCholesky',  @acc_rpcholesky,   '#DC267F', '^', '-.'};
nmeth = size(methods,1);
nk    = numel(ks);

K = KernelOracle(Xtr, sigma);

for i = 1:nmeth                                 % untimed warm-up
    methods{i,2}(K, 20);
end

ent = zeros(nmeth, nk, trials);
tim = zeros(nmeth, nk, trials);
err = zeros(nmeth, nk, trials);
tre = zeros(nmeth, nk, trials);
nsel = zeros(nmeth, nk, trials);

wall = tic;
for j = 1:nk
    for t = 1:trials
        for i = randperm(nmeth)
            K.reset();
            t0 = tic;
            [S, C] = methods{i,2}(K, ks(j));
            tim(i,j,t)  = toc(t0);
            ent(i,j,t)  = K.count;
            nsel(i,j,t) = numel(S);

            tre(i,j,t) = nystrom_trace_error(C, S, ones(ntr,1));    % diag(A) = 1

            beta = nystrom_krr(C, S, ytr, lam);
            ft   = gauss_kernel(Xte, Xtr(S,:), sigma) * beta;
            err(i,j,t) = mean(sign(ft) ~= yte);
        end
    end
    fprintf('  k = %4d done, %7.1f s elapsed\n', ks(j), toc(wall));
end
fprintf('Sweep finished in %.1f s\n', toc(wall));

if any(nsel(:) ~= reshape(repmat(ks, nmeth, 1, trials), [], 1))
    warning('column_nystrom_experiment:FewerPivots', ...
            'Some runs returned fewer than k landmarks (residual exhausted).');
end

datadir = fullfile(here, '..', 'data');
if ~isfolder(datadir), mkdir(datadir); end
save(fullfile(datadir, 'column_nystrom_results.mat'), ...
     'ks', 'trials', 'methods', 'ent', 'tim', 'err', 'tre', 'nsel', 'lam', 'sigma', 'valmse', 'grid_lam');

%% Summary table

q = @(A) quantile(A, [0.10 0.50 0.90], 3);
fprintf('\nMedians over %d trials\n', trials);
fprintf('%-24s %6s %14s %12s %12s %12s\n', 'method', 'k', 'entries', 'time (s)', 'trace err', 'test err');
for i = 1:nmeth
    for j = 1:nk
        fprintf('%-24s %6d %14.4g %12.4g %12.4e %12.4f\n', methods{i,1}, ks(j), ...
                median(ent(i,j,:)), median(tim(i,j,:)), median(tre(i,j,:)), median(err(i,j,:)));
    end
end

%% Plot: 2-by-2 grid
%   rows:    relative trace error (top), test misclassification (bottom)
%   columns: against k (left), against runtime (right)
% Each runtime panel plots the median error at each k against the median
% runtime at that k, with the error's 10/90 band.

fig_w     = 1400;
ax_left   = [150 790];
ax_width  = 520;
ax_height = 330;
ax_bottom = [690 220];          % top row, bottom row
fig_h     = ax_bottom(1) + ax_height + 70;

Tq = q(tre);   Eq = q(err);   Rq = q(tim);
rows = { Tq, 'Relative trace error',     'log'; ...
         Eq, 'Test misclassification',   'linear'};

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h], ...
             'PaperPositionMode','auto');

for r = 1:2
    Q = rows{r,1};
    for c = 1:2
        ax = axes('Parent',fig,'Units','pixels', ...
                  'Position',[ax_left(c) ax_bottom(r) ax_width ax_height]);
        hold(ax,'on')
        for i = 1:nmeth
            if c == 1
                x = ks;
            else
                x = Rq(i,:,2);
            end
            plot_shaded(x, Q(i,:,2), Q(i,:,1), Q(i,:,3), methods{i,3}, ...
                        'LineWidth', 1.5, 'MarkerSize', 6, ...
                        'Marker', methods{i,4}, 'LineStyle', methods{i,5});
        end
        set(ax, 'XScale', 'log', 'YScale', rows{r,3})
        grid(ax,'on')
        if c == 1
            xlim(ax, [min(ks) max(ks)]);
            ylabel(ax, rows{r,2}, 'Interpreter','latex')
        else
            xlim(ax, [min(Rq(:,:,2),[],'all') max(Rq(:,:,2),[],'all')]);
        end
        if r == 2
            if c == 1
                xlabel(ax, 'Number of landmarks $k$', 'Interpreter','latex')
            else
                xlabel(ax, 'Runtime (s)', 'Interpreter','latex')
            end
        end
    end
end

lines = flipud(findobj(ax,'Type','Line'));
lgd = legend(lines, methods(:,1), 'Interpreter','latex', ...
             'NumColumns', nmeth, 'Units','pixels');
lgd.Position(1) = max(0, (fig_w - lgd.Position(3))/2);
lgd.Position(2) = 40;

ansave(fig, fullfile(here,'..','..','figures','column_nystrom'));

%% Local functions

function [Xtr, ytr, Xte, yte] = encode_adult(Ttr, Tte)
%ENCODE_ADULT  z-score numeric features, one-hot categorical features.
%   Statistics and category lists come from the training table only.

numv = {'age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week'};
catv = {'workClass','education','marital_status','occupation', ...
        'relationship','race','sex','native_country'};

Ntr = Ttr{:,numv};  Nte = Tte{:,numv};
mu  = mean(Ntr);    sd  = std(Ntr);
Ntr = (Ntr - mu) ./ sd;
Nte = (Nte - mu) ./ sd;

Ctr = [];  Cte = [];
for v = catv
    a = Ttr.(v{1});  b = Tte.(v{1});
    cats = categories(a);
    Za = zeros(numel(a), numel(cats)+1);
    Zb = zeros(numel(b), numel(cats)+1);
    sa = string(a);  sb = string(b);
    for c = 1:numel(cats)
        Za(:,c) = sa == string(cats{c});
        Zb(:,c) = sb == string(cats{c});
    end
    Za(:,end) = ismissing(sa);                  % missing as its own category
    Zb(:,end) = ismissing(sb);
    Ctr = [Ctr Za];                             %#ok<AGROW>
    Cte = [Cte Zb];                             %#ok<AGROW>
end

Xtr = [Ntr Ctr];
Xte = [Nte Cte];
ytr = 2*contains(string(Ttr.salary), '>50K') - 1;
yte = 2*contains(string(Tte.salary), '>50K') - 1;

end
