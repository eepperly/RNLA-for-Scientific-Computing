%% Interpolative decomposition: six index-selection methods
%
% TEST MATRIX.  The `kernel` example of section 7.3 of
%
%   E. N. Epperly, "Adaptive randomized pivoting and volume sampling" (2025),
%
% which tabulates inverse distances between two point sets,
%
%     A(i,j) = 1 / ||x_i - y_j||,     i,j = 1,...,n,   n = 10^4,
%
% with {x_i} an equispaced Cartesian grid on [0,1)^2 and {y_j} an equispaced
% grid on [1,2)x[0,1).  The two grids are disjoint, so the kernel is never
% singular, and the matrix is the kind of well-separated off-diagonal block that
% arises in rank-structured matrix methods.  Its singular values decay
% geometrically but far more slowly than a smooth analytic kernel, which is what
% makes a sweep out to k = 400 meaningful.
%
% A is NOT symmetric here: the row and column index sets live on different
% grids.  The rangefinder is therefore built as (Om'*A)' = A'*Om, a basis for
% the row space, which is what a column ID needs.
%
% METHODS.  All select k indices I:
%
%   CPQR                 Golub-Businger column-pivoted QR, stopped at k pivots
%                        (CPQR_PARTIAL, written out rather than MATLAB's qr).
%   ARP                  Adaptive randomized pivoting: sketch, rangefinder
%                        Q = orth(A'*Om) with k columns, then randomly
%                        pivoted QR on Q' by the direct implementation.
%   Accelerated RPQR     Randomly pivoted QR applied to A itself, by rejection
%                        sampling in blocks.  Same distribution as the direct
%                        version, verified by a two-sample test.
%   CPQR on a sketch     MATLAB's qr on the 2k-by-n sketch Om'*A, keeping the
%                        first half of the pivots.
%   Leverage             Leverage score sampling on the rangefinder basis Q.
%   Uniform              k indices uniformly at random.
%
% Sketches are SparseStack with zeta = 4 and an oversampling factor of two, so
% every sketch is 2k wide.
%
% COLUMN INTERPOLATIVE DECOMPOSITION.  Every method computes
%
%     A ~ A(:,I) * Z,
%
% so I indexes columns of A and Z is k-by-n.  Exact (solid lines), from
% A(:,I) = Q*R:
%
%     Z = R \ (Q'*A),
%
% which is pinv(A(:,I))*A, the projection of A onto the column space of A(:,I).
% Oversampled (OSID, dashed lines), with Om an n-by-2k SparseStack:
%
%     Z = (Om'*A(:,I)) \ (Om'*A).
%
% OSID is only run for ARP, CPQR on a sketch, leverage and uniform.
%
% ARP and the leverage rule build their rangefinder from A'*Om, a basis for the
% row space, which is what the column ID needs.

n      = 1e4;                        % matrix dimension
zeta   = 4;                          % SparseStack sparsity
os     = 2;                          % sketch oversampling factor
ks     = 20:20:400;
trials = 100;                        % random trials

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'), fullfile(here,'..','sketching'));

datadir = fullfile(here,'..','data');
if ~isfolder(datadir), mkdir(datadir); end
cachefile = fullfile(datadir,'id_experiment.mat');
params = struct('n',n,'zeta',zeta,'os',os,'ks',ks,'trials',trials);

%          name                 color      mark  OSID
methods = { ...
    'CPQR',              '#000000', 'o', false; ...
    'ARP',               '#DC267F', 's', true;  ...
    'Accelerated RPQR',  '#009E73', '^', false; ...
    'CPQR on a sketch',  '#648FFF', 'd', true;  ...
    'Leverage',          '#FE6100', 'v', true;  ...
    'Uniform',           '#785EF0', '>', true};
nmeth = size(methods,1);
nk    = numel(ks);

%% Test matrix

ng = round(sqrt(n));
fprintf('Building %d-by-%d kernel matrix (%.1f GB)...\n', n, n, n^2*8/1e9);
tb = tic;
t  = (0:ng-1)/ng;
[X1,X2] = ndgrid(t, t);                  % source grid on [0,1)^2
[Y1,Y2] = ndgrid(1+t, t);                % target grid on [1,2)x[0,1)
A = 1 ./ pdist2([X1(:) X2(:)], [Y1(:) Y2(:)]);
nrmA = norm(A,'fro');
fprintf('  %.1f s, ||A||_F = %.4e, max entry = %.3f\n', toc(tb), nrmA, max(A(:)));

%% Sweep

rebuild = true;
if isfile(cachefile)
    cached = load(cachefile,'params');
    if isequal(cached.params, params)
        fprintf('Loading results from cache\n');
        load(cachefile,'errX','errO','timX','timO');
        rebuild = false;
    else
        fprintf('Cache built with different parameters; rebuilding.\n');
    end
end

if rebuild
    errX = nan(nmeth, nk, trials);   timX = nan(nmeth, nk, trials);   % exact ID
    errO = nan(nmeth, nk, trials);   timO = nan(nmeth, nk, trials);   % OSID

    wall = tic;
    for jk = 1:nk
        k = ks(jk);
        for i = 1:nmeth
            nrep = trials;
            if i == 1, nrep = 1; end             % CPQR is deterministic
            for t = 1:nrep
                t0 = tic;
                I  = select_indices(A, n, k, i, zeta, os);
                tsel = toc(t0);

                t0 = tic;
                Z = exact_id(A, I);
                timX(i,jk,t) = tsel + toc(t0);
                errX(i,jk,t) = id_error(A, Z, I, nrmA);

                if methods{i,4}
                    t0 = tic;
                    Zo = osid(A, I, os*k, zeta);
                    timO(i,jk,t) = tsel + toc(t0);
                    errO(i,jk,t) = id_error(A, Zo, I, nrmA);
                end
            end
            if nrep == 1                          % broadcast the deterministic run
                errX(i,jk,:) = errX(i,jk,1);  timX(i,jk,:) = timX(i,jk,1);
            end
        end
        fprintf('  k = %3d done, %.1f s elapsed\n', k, toc(wall));
    end
    fprintf('Sweep finished in %.1f s\n', toc(wall));
    save(cachefile,'errX','errO','timX','timO','params','-v7.3');
end

eX = median(errX,3);  eXl = min(errX,[],3);  eXh = max(errX,[],3);
eO = median(errO,3);  eOl = min(errO,[],3);  eOh = max(errO,[],3);
tX = median(timX,3);  tXl = min(timX,[],3);  tXh = max(timX,[],3);
tO = median(timO,3);  tOl = min(timO,[],3);  tOh = max(timO,[],3);

fprintf('\n%-20s %12s %12s %12s\n','method','err k=20','err k=40','time k=40');
for i = 1:nmeth
    fprintf('%-20s %12.3e %12.3e %12.3f\n', methods{i,1}, ...
            eX(i,ks==20), eX(i,ks==40), tX(i,ks==40));
end

%% Figure: accuracy and runtime against k, side by side

fig_w = 1750; fig_h = 760;
ax_x  = [150 1005];  ax_w = 630;
ax_y  = 235;         ax_h = 400;

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h], ...
             'PaperPositionMode','auto');

for which = 1:2
    ax = axes('Parent',fig,'Units','pixels', ...
              'Position',[ax_x(which) ax_y ax_w ax_h]);
    hold(ax,'on')
    for i = 1:nmeth
        if which == 1
            plot_shaded(ks, eX(i,:), eXl(i,:), eXh(i,:), methods{i,2}, ...
                'LineWidth',1.5,'MarkerSize',6,'Marker',methods{i,3},'LineStyle','-');
        else
            plot_shaded(ks, tX(i,:), tXl(i,:), tXh(i,:), methods{i,2}, ...
                'LineWidth',1.5,'MarkerSize',6,'Marker',methods{i,3},'LineStyle','-');
        end
    end
    for i = 1:nmeth
        if ~methods{i,4}, continue, end
        if which == 1
            plot_shaded(ks, eO(i,:), eOl(i,:), eOh(i,:), methods{i,2}, ...
                'LineWidth',1.5,'MarkerSize',6,'Marker',methods{i,3},'LineStyle','--');
        else
            plot_shaded(ks, tO(i,:), tOl(i,:), tOh(i,:), methods{i,2}, ...
                'LineWidth',1.5,'MarkerSize',6,'Marker',methods{i,3},'LineStyle','--');
        end
    end
    set(ax,'YScale','log')
    xlabel(ax,'Index set size $k$','Interpreter','latex')
    if which == 1
        ylabel(ax,['$\|\mbox{\boldmath $A$}-\widehat{\mbox{\boldmath $A$}}\|_F' ...
                   '/\|\mbox{\boldmath $A$}\|_F$'],'Interpreter','latex')
        % title(ax,'Interpolative decomposition accuracy','Interpreter','latex')
    else
        ylabel(ax,'Total runtime (s)','Interpreter','latex')
        % title(ax,'Total runtime (selection and ID)','Interpreter','latex')
    end
    xlim(ax,[min(ks) max(ks)]); grid(ax,'on')
    ax.Toolbar.Visible = 'off';

    % Build the legend on the left panel, where every method has a line, and
    % position it once both axes exist.
    if which == 1
        lines = flipud(findobj(ax,'Type','Line'));
        lgd = legend(lines(1:nmeth), methods(:,1), 'Interpreter','latex', ...
                     'Units','pixels','NumColumns',3);
    end
end

lgd.Position(1) = max(0,(fig_w - lgd.Position(3))/2);
lgd.Position(2) = 40;

ansave(fig, fullfile(here,'..','..','figures','id_comparison'));

%% Local functions

function I = select_indices(A, n, k, which, zeta, os)
%SELECT_INDICES  k indices of A by the requested rule.
switch which
    case 1                                   % CPQR, Golub-Businger
        I = cpqr_partial(A, k);
    case 2                                   % ARP: rangefinder then direct RPQR
        Q = rangefinder(A, n, k, zeta);
        I = RandomlyPivotedQR(Q', k);
    case 3                                   % accelerated RPQR, on A itself
        I = AcceleratedRPQR(A, k);
    case 4                                   % CPQR on a sketch
        [~,~,Omt] = SparseStack(zeros(n,0), [], os*k, zeta);
        [~,~,p]   = qr(Omt*A, 'vector');
        I = p(1:k);                          % first half of the 2k pivots
    case 5                                   % leverage scores on the rangefinder
        Q = rangefinder(A, n, k, zeta);
        I = SelectLeverage(Q, k);
    case 6                                   % uniform
        I = randperm(n, k);
end
I = sort(I(:));
end

function Q = rangefinder(A, n, k, zeta)
%RANGEFINDER  Orthonormal basis for the row space of A from a width-k sketch.
%   Y = (Om'*A)' = A'*Om, so Q spans the row space, which is what a column ID
%   needs.  The sketch is k wide, not os*k: the oversampling factor is applied
%   to the OSID and to CPQR-on-a-sketch, but the rangefinder here matches the
%   k columns that Algorithm 2.1 of the ARP paper uses.
[~,~,Omt] = SparseStack(zeros(n,0), [], k, zeta);
Y = (Omt * A)';
[Q,~] = qr(Y,"econ");
end

function Z = exact_id(A, I)
%EXACT_ID  Z = R \ (Q'*A) from A(:,I) = Q*R, so that A ~ A(:,I)*Z.
%   lsqminnorm rather than backslash for the triangular solve: as k approaches
%   the numerical rank R becomes singular, and the minimum-norm solution is
%   what pinv(A(:,I))*A means there.  The two agree whenever R is invertible.
[Q, R] = qr(A(:,I), 0);
Z = lsqminnorm(R, Q'*A);
end

function Z = osid(A, I, width, zeta)
%OSID  Z = (Om'*A(:,I)) \ (Om'*A) with Om an n-by-width SparseStack.
%   lsqminnorm rather than backslash: near the numerical rank the sketched
%   submatrix is rank deficient and backslash warns and returns a large
%   solution, where the minimum-norm one is what the pseudoinverse means.
[~,~,Omt] = SparseStack(zeros(size(A,1),0), [], width, zeta);
Z = lsqminnorm(Omt*A(:,I), Omt*A);
end

function e = id_error(A, Z, I, nrmA)
%ID_ERROR  ||A - A(:,I)*Z||_F / ||A||_F, in column blocks to avoid an n-by-n
%   temporary (800 MB for the test matrix here).
n = size(A,2);
C = A(:,I);
s2 = 0;
blk = 1000;
for lo = 1:blk:n
    hi = min(lo+blk-1, n);
    R  = A(:,lo:hi) - C * Z(:,lo:hi);
    s2 = s2 + sum(R(:).^2);
end
e = sqrt(s2) / nrmA;
end
