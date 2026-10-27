%% Low-rank approximation of the inverse 5-point Laplacian
%
% A = L^{-1}, where L is the standard five-point Laplacian with homogeneous
% Dirichlet boundary conditions on an n1-by-n1 interior grid, so A is N-by-N
% with N = n1^2 = 4*10^4.  A is dense and is never formed: it is applied through
% a sparse Cholesky factor of L, which gives the implicit X -> A*X interface
% the Nystrom routines take.
%
% ORDERING: the factorization uses dissect(), MATLAB's built-in METIS nested
% dissection, rather than the AMD ordering chol() picks on its own.  Nested
% dissection is the right ordering for a grid Laplacian and measures 14% less
% fill at n1 = 200 and 18% less at n1 = 400, the gap widening with the grid.
%
% A inherits the eigenvectors of L and inverts its eigenvalues, so its leading
% eigenvectors are the smooth low-frequency grid modes and its spectrum decays
% slowly -- the regime where subspace iteration and block Krylov earn their
% extra passes over a single-pass sketch.
%
% Left panel:  spectral-norm error of the approximation truncated to rank
%              blk = 30, against the best rank-30 error.  The untruncated
%              variants keep every component they build, so they are not
%              restricted to rank 30.
% Right panel: accuracy of the 20th eigenvector against an eigs reference.
% Both against the number of matrix-vector products, the resource each method
% actually spends.
%
% CACHING: the Cholesky factor, its permutation, and the reference eigenpairs
% are all cached in ../data/.  An explicit chol() is used rather than
% decomposition() because MATLAB refuses to serialize a decomposition object
% ("Loading a decomposition is not supported.  The loaded object is the
% decomposition of an empty matrix", MATLAB:decomposition:LoadNotSupported).

n1     = 200;          % interior grid points per dimension; N = n1^2
rank_r = 20;           % truncation rank, and the eigenvector index of interest
blk    = 30;           % block size for subspace iteration and RBKI
qmax   = 10;           % deepest iteration; matvecs run to (qmax+1)*blk
trials = 100;          % repetitions, for the median and 10/90 bands

N = n1^2;
h = 1/(n1+1);

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'));

datadir = fullfile(here,'..','data');
if ~isfolder(datadir)
    mkdir(datadir);
end
cachefile = fullfile(datadir, sprintf('nystrom_laplace_%d.mat', n1));
params    = struct('n1',n1,'rank_r',rank_r,'blk',blk);   % blk sets nev

%% Operator and reference eigenpairs, both cached

nev = max(rank_r, blk) + 5;   % enough for the rank_r cluster and for dref(blk+1)

rebuild = true;
if isfile(cachefile)
    cached = load(cachefile,'params');
    if isequal(cached.params, params)
        fprintf('Loading Cholesky factor and reference eigenpairs from cache\n');
        load(cachefile,'R','perm','Vref','dref');
        rebuild = false;
    else
        fprintf('Cache built with different parameters; rebuilding.\n');
    end
end

if rebuild
    e = ones(n1,1);
    T = spdiags([-e 2*e -e], -1:1, n1, n1) / h^2;
    L = kron(speye(n1),T) + kron(T,speye(n1));

    fprintf('Factoring L (%d-by-%d) with nested dissection...\n', N, N);
    tc   = tic;
    perm = dissect(L);                      % METIS nested dissection ordering
    R    = chol(L(perm,perm));              % R'*R = L(perm,perm)
    fprintf('  chol: %.1f s, nnz(R) = %.2e (%.1fx nnz(L))\n', ...
            toc(tc), nnz(R), nnz(R)/nnz(L));

    Aref = @(X) laplace_solve(R, perm, X);
    fprintf('Computing %d reference eigenpairs with eigs...\n', nev);
    te = tic;
    [Vref, Dref] = eigs(Aref, N, nev, 'largestabs', ...
                        'IsFunctionSymmetric', true, 'Tolerance', 1e-15);
    [dref, ord] = sort(diag(Dref),'descend');
    Vref = Vref(:,ord);
    fprintf('  eigs: %.1f s\n', toc(te));

    save(cachefile,'R','perm','Vref','dref','params','-v7.3','-nocompression');
end

Afun = @(X) laplace_solve(R, perm, X);

% The Dirichlet Laplacian on a square has exactly degenerate eigenvalues: the
% grid modes (i,j) and (j,i) share an eigenvalue whenever i ~= j.  Only the
% diagonal modes (i,i) are simple, which for this ordering means indices
% 1, 4, 11, 20, ...  Index 20 is the mode (4,4) and IS simple, so the 20th
% eigenvector is well defined and the comparison below is meaningful.  The
% multiplicity is still computed and asserted rather than assumed, and the
% accuracy is measured as the angle to the eigenspace of that eigenvalue,
% which for a simple eigenvalue is the usual angle between two vectors.
clust = abs(dref - dref(rank_r)) <= 1e-8 * abs(dref(rank_r));
Vc    = Vref(:, clust);
fprintf('Eigenvalue %d has multiplicity %d (relative gap to %d: %.2e)\n', ...
        rank_r, size(Vc,2), rank_r+1, abs(dref(rank_r)-dref(rank_r+1))/dref(rank_r));
if size(Vc,2) ~= 1
    warning('nystrom_experiment:Degenerate', ...
            ['Eigenvalue %d is degenerate (multiplicity %d); the right panel ' ...
             'measures the angle to its eigenspace, not to a single vector.'], ...
            rank_r, size(Vc,2));
end

best = dref(blk+1) / dref(1);           % optimal relative rank-blk error, to
                                        % match what the left panel truncates to

%% Sweep: error and eigenvector accuracy against matvec count

qs      = 0:qmax;
matvecs = (qs+1)*blk;
npt     = numel(qs);

% A method and its untruncated variant share a color and a marker, and are told
% apart by the line style: dashed is truncated, solid is not.  Subspace
% iteration returns exactly rank blk, the truncation rank, so its truncation
% removes nothing and it counts as untruncated -- flagging it false here is a no-op
% numerically.  The eigenvector panel plots only the untruncated curves, which
% leaves one per method.
%          name                   sketch                                      color      mark style truncate
methods = { ...
    'Nystrom',               @(q) Nystrom(Afun,(q+1)*blk,N),            '#648FFF','o','--', true;  ...
    'Nystrom (untruncated)', @(q) Nystrom(Afun,(q+1)*blk,N),            '#648FFF','o','-',  false; ...
    'Subspace iteration',    @(q) NystromSubspace(Afun,blk,q,N),        '#DC267F','d','-',  false; ...
    'RBKI',                  @(q) NystromKrylov(Afun,blk,q,blk,N),      '#009E73','^','--', true;  ...
    'RBKI (untruncated)',    @(q) NystromKrylov(Afun,blk,q,[],N),       '#009E73','^','-',  false};
nmeth = size(methods,1);

err = zeros(nmeth, npt, 3);
ang = zeros(nmeth, npt, 3);

wall = tic;
for i = 1:nmeth
    f = methods{i,2};
    for j = 1:npt
        te = zeros(trials,1);
        ta = zeros(trials,1);
        for t = 1:trials
            [U, Lam] = f(qs(j));

            % Left panel: the truncated methods are cut to the block size, so
            % every truncated curve reports a rank-blk approximation whatever
            % the method built.  A - Ut*diag(Lt)*Ut' is PSD, so its largest
            % eigenvalue is the spectral norm; A is implicit, so use eigs.
            if methods{i,6}
                rt = min(blk, numel(Lam));
                Ut = U(:,1:rt);
                Lt = Lam(1:rt);
            else
                Ut = U;
                Lt = Lam;
            end
            res = @(x) Afun(x) - Ut*(Lt .* (Ut'*x));
            te(t) = abs(eigs(res, N, 1, 'largestabs', ...
                             'IsFunctionSymmetric', true, 'Tolerance', 1e-8)) / dref(1);

            % Right panel: angle from the rank_r-th eigenvector of the full,
            % untruncated factor to the true eigenspace.  Kept separate from
            % the truncation above so the two panels cannot interfere.
            v     = U(:,rank_r);
            ta(t) = norm(v - Vc*(Vc'*v));
        end
        err(i,j,:) = quantile(te, [0.10 0.50 0.90]);
        ang(i,j,:) = quantile(ta, [0.10 0.50 0.90]);
    end
    fprintf('  %-22s done, %.1f s elapsed\n', methods{i,1}, toc(wall));
end
fprintf('Sweep finished in %.1f s\n', toc(wall));

%% Plot

fig = figure('Units','normalized','Position',[0.10 0.25 0.75 0.42], ...
             'PaperPositionMode','auto');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(1); hold on
for i = 1:nmeth
    plot_shaded(matvecs, err(i,:,2), err(i,:,1), err(i,:,3), ...
                methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
% Drawn with plot rather than yline so it becomes a Line and joins the legend.
plot(matvecs, best*ones(size(matvecs)), 'k:', 'LineWidth', 1.5);
set(ax1,'YScale','log')
xlabel('Matrix--vector products','Interpreter','latex')
ylabel('$\|\mbox{\boldmath $A$} - \widehat{\mbox{\boldmath $A$}}\|/\|\mbox{\boldmath $A$}\|$', ...
       'Interpreter','latex')
title('Approximation error','Interpreter','latex')
xlim([min(matvecs) max(matvecs)]); grid on

ax2 = nexttile(2); hold on
for i = find(~[methods{:,6}])       % untruncated curves only
    plot_shaded(matvecs, ang(i,:,2), ang(i,:,1), ang(i,:,3), ...
                methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
set(ax2,'YScale','log')
xlabel('Matrix--vector products','Interpreter','latex')
ylabel('$\sin\angle(\widehat{\mbox{\boldmath $u$}}_{20}, \mbox{\boldmath $u$}_{20})$', ...
       'Interpreter','latex')
title('Accuracy of the 20th eigenvector','Interpreter','latex')
xlim([min(matvecs) max(matvecs)]); grid on

% Panel 1 carries the reference line too, so build the legend from it.  Lines
% come back newest first, so flipud puts them in creation order: the five
% methods, then the reference line.
lines = flipud(findobj(ax1,'Type','Line'));

% Each method appears once.  A truncated curve shares its color and marker with
% its untruncated partner, so the solid handle stands for the pair and the line
% style alone distinguishes them in the plot.
sel    = [find(~[methods{:,6}]), nmeth+1];
labels = {'Nystr\"om', 'NysSI', 'NysBKI', 'Optimal'};
lgd = legend(lines(sel), labels, 'Interpreter','latex', 'NumColumns', 4);
lgd.Layout.Tile = 'south';

ansave(fig, fullfile(here,'..','..','figures','nystrom_laplace'));

%% Local functions

function Y = laplace_solve(R, perm, X)
%LAPLACE_SOLVE  Apply L^{-1} through the cached Cholesky factor.
Y = zeros(size(X));
Y(perm,:) = R \ (R' \ X(perm,:));
end
