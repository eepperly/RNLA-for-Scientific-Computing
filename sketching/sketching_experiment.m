%% Sketching experiment, part 1: build the test matrix A = L*B
%
% L is the standard five-point discretization of the 2D advection-diffusion
% problem on the unit square with homogeneous Dirichlet boundary conditions,
%
%       -Lap u + c*(du/dx + du/dy) = f  on (0,1)^2,   u = 0 on the boundary,
%       f(x,y) = sin(x)*cos(y),
%
% on an n1-by-n1 interior grid, so L is N-by-N with N = n1^2 = 10^6.  The
% advection term uses centered differences, which is the appropriate choice
% in the small grid Peclet regime used here (Pe = c*h/2 = 0.1, so the problem
% is diffusion dominated and no upwinding is needed).
%
% B is a Krylov basis for (L,b) built by Arnoldi with *partial*
% orthogonalization: each new vector is orthogonalized against only the
% previous k = 8 basis vectors.  A = L*B is then N-by-m = 10^6-by-100.
%
% The advection term is what makes the truncation meaningful.  For the pure
% Poisson problem L is symmetric, Arnoldi is Lanczos, and the exact recurrence
% is already three-term, so truncating to the last k vectors is mathematically
% equivalent to full orthogonalization and B comes out orthonormal.  With
% advection L is nonsymmetric, the truncation genuinely loses orthogonality,
% and B is ill conditioned -- which is the point of the test.
%
% A is cached to data/ after it is built.  Rerunning the script reloads A
% instead of regenerating it, unless the parameters below have changed.  The
% sketching routines alongside this script and utils/ are put on the path
% here so that part 2 can use them.
%
% WARNING: A and B are each 10^6-by-100 dense doubles, i.e. 0.8 GB apiece, and
% the cache file is another 0.8 GB on disk.  Peak memory during the
% conditioning diagnostics is a few GB.  Reduce n1 and/or m if that is too
% much.

n1 = 1000;          % interior grid points per dimension; N = n1^2
m  = 100;           % Krylov dimension = number of columns of A
k  = 8;             % partial orthogonalization depth
Pe = 0.1;           % grid Peclet number c*h/2 (small => diffusion dominated)

N = n1^2;
h = 1/(n1+1);
c = 2*Pe/h;         % advection speed in each coordinate direction

%% Paths and cache location

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'));

datadir = fullfile(here,'..','data');
if ~isfolder(datadir)
    mkdir(datadir);
end

params    = struct('n1',n1,'m',m,'k',k,'Pe',Pe);
cachefile = fullfile(datadir,'sketching_testmat.mat');

%% Load A from cache if it matches the current parameters, else rebuild

rebuild = true;
if isfile(cachefile)
    cached = load(cachefile,'params');
    if isequal(cached.params, params)
        fprintf('Loading A from %s\n', cachefile);
        load(cachefile,'A','diagnostics');
        rebuild = false;
    else
        fprintf('Cache in %s was built with different parameters; rebuilding.\n', cachefile);
    end
end

if rebuild

    % L: five-point Laplacian plus centered advection, Dirichlet BCs
    e  = ones(n1,1);
    T  = spdiags([-e 2*e -e], -1:1, n1, n1) / h^2;   % 1D Dirichlet Laplacian
    D  = spdiags([-e 0*e e], -1:1, n1, n1) / (2*h);  % 1D centered difference
    I1 = speye(n1);
    L  = kron(I1,T) + kron(T,I1) + c*(kron(I1,D) + kron(D,I1));

    % Right-hand side f(x,y) = sin(x)*cos(y) on the interior grid
    xg = (1:n1)' * h;
    [X,Y] = ndgrid(xg,xg);
    f = sin(X) .* cos(Y);
    f = f(:);
    b = f / norm(f);                                % Arnoldi needs a unit start

    % Partially orthogonalized Arnoldi: build B, and A = L*B along the way
    B = zeros(N,m);
    A = zeros(N,m);

    B(:,1) = b;
    for j = 1:m
        w = L*B(:,j);
        A(:,j) = w;                                 % jth column of A = L*B
        if j == m
            break
        end
        for i = max(1,j-k+1):j                      % last k vectors only
            w = w - (B(:,i)'*w) * B(:,i);
        end
        nw = norm(w);
        if nw < eps*norm(A(:,j))
            error('Arnoldi broke down at step %d', j);
        end
        B(:,j+1) = w / nw;
    end

    % Conditioning diagnostics.  These are full SVDs of tall-skinny matrices
    % and are the expensive part of the script; B is cleared before cond(A)
    % to keep the peak memory down.
    diagnostics.condL = condest(L);
    diagnostics.orthB = norm(B'*B - eye(m));
    diagnostics.condB = cond(B);
    clear B
    diagnostics.condA = cond(A);
    diagnostics.normA = norm(A,'fro');

    fprintf('Saving A to %s\n', cachefile);
    save(cachefile,'A','params','diagnostics','-v7.3','-nocompression');

end

%% Report

fprintf('A is %d-by-%d\n', size(A,1), size(A,2));
fprintf('cond(L)     = %.3e\n', diagnostics.condL);
fprintf('cond(B)     = %.3e\n', diagnostics.condB);
fprintf('cond(A)     = %.3e\n', diagnostics.condA);
fprintf('||B''B - I|| = %.3e\n', diagnostics.orthB);
fprintf('||A||_F     = %.3e\n', diagnostics.normA);

%% Part 2: compare sketching matrices

% Each method is timed and scored on two orthonormal test bases: the Krylov
% basis orth(A), and the maximally coherent Q = [I; 0].  sigma_min(Om'*Q) is
% the quantity that matters -- Om embeds range(Q) iff it stays away from 0.
%
% WARNING: this is the expensive half of the script.  Cost is dominated by
% Gaussian, at O(k*N*m) flops per trial with no fast transform, and it is
% incurred trials*numel(ks)*2 times.  Cut `trials` or drop Gaussian from the
% list for a quick look.

trials = 100;
m      = 100;
ks     = round(m * [1.25 1.5 2 3 4 6 8 10]);      % sketch sizes, relative to the m columns

methods = { ...
%    name                      sketch                            color      marker  style
    'Gaussian',                @(Q,k) Gaussian(Q,[],k),         '#648FFF', 'o',    '-'; ...
    'CountSketch',             @(Q,k) SparseStack(Q,[],k,1),    '#785EF0', 's',    '--'; ...
    'SparseStack ($\zeta=4$)', @(Q,k) SparseStack(Q,[],k,4),    '#DC267F', 'd',    ':'; ...
    'SRFT (1 round)',          @(Q,k) SRFT(Q,[],k,1),           '#FE6100', '^',    '-.'; ...
    'SRFT (2 rounds)',         @(Q,k) SRFT(Q,[],k,2),           '#FFB000', 'v',    '-'; ...
    'Uniform sampling',        @(Q,k) UniformSampling(Q,[],k),  '#009E73', 'p',    '--'; ...
    'Leverage sampling',       @(Q,k) LeverageSampling(Q,[],k), '#000000', 'h',    ':'; ...
    'LASO',                    @(Q,k) LESS(Q,[],k),             '#56B4E9', '>',    '-.'; ...
    'Multisketch',             @(Q,k) Multisketch(Q,[],k),      '#7E2F8E', '<',    '-'};

nmeth = size(methods,1);
nk    = numel(ks);

% Indexed [method, k, quantile (10/50/90), test basis (1 = orth(A), 2 = [I;0])]
times = zeros(nmeth, nk, 3, 2);
smins = zeros(nmeth, nk, 3, 2);

wall = tic;
for q = 1:2
    if q == 1
        fprintf('Building Q = orth(A)...\n');
        [Qm,~] = qr(A(:,1:m),"econ");
        clear A     % 4 GB; rerun the script to reload it from the cache
    else
        fprintf('Building Q = [I; 0]...\n');
        Qm = zeros(N, m);
        Qm(1:m,1:m) = eye(m);
    end
    fprintf('  panel %d: %-24s %6.1f s elapsed\n', q, 'done orthogonalizing', toc(wall));

    for i = 1:nmeth
        f = methods{i,2};
        for j = 1:nk
            tt = zeros(trials,1);
            ss = zeros(trials,1);
            for t = 1:trials
                tic;
                SQ    = f(Qm, ks(j));
                tt(t) = toc;
                ss(t) = min(svd(full(SQ)));
            end
            times(i,j,:,q) = quantile(tt, [0.10 0.50 0.90]);
            smins(i,j,:,q) = quantile(ss, [0.10 0.50 0.90]);
        end
        fprintf('  panel %d: %-24s %6.1f s elapsed\n', q, methods{i,1}, toc(wall));
    end
    clear Qm
end
fprintf('Comparison finished in %.1f s\n', toc(wall));

%% Plot: one figure per test basis, runtime on the left, sigma_min on the right

basis_name = {'Arnoldi', 'Coherent'};

% Explicit axes positions rather than a tiled layout.  The two figures carry
% different furniture -- titles on the Arnoldi one, a legend on the Coherent
% one -- so a layout manager would give them different plot heights.  Fixing
% the geometry by hand keeps the plot boxes identical: the Coherent figure is
% taller by legend_h, and its axes take a proportionally smaller normalized
% height so the absolute box is the same size in both.
fig_w     = 0.75;           % figure width, normalized to the screen
fig_h     = 0.40;           % height of the Arnoldi figure
legend_h  = 0.12;           % extra height given to the Coherent figure
ax_left   = [0.085 0.575];  % left edge of each axes
ax_width  = 0.360;          % trimmed to leave room for the long right title
ax_bottom = 0.225;          % axes bottom and height, as fractions of fig_h
ax_height = 0.640;          % raised bottom leaves blank space under the x label
smins(smins == 0) = eps;

for q = 1:2
    extra = (q == 2) * legend_h;
    H     = fig_h + extra;

    % PaperPositionMode 'auto' makes saveas export what is on screen; the
    % default fits the figure to letter paper, which is what crops the export.
    fig = figure('Units','normalized','Position',[0.10 0.25 fig_w H], ...
                 'PaperPositionMode','auto');

    % Scale out the extra height so both figures get the same absolute box.
    hN = ax_height * fig_h / H;
    bN = (ax_bottom * fig_h + extra) / H;

    ax1 = axes('Parent',fig,'Position',[ax_left(1) bN ax_width hN]);
    hold(ax1,'on')
    for i = 1:nmeth
        plot_shaded(ks / m, times(i,:,2,q), times(i,:,1,q), times(i,:,3,q), ...
                    methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                    'Marker', methods{i,4}, 'LineStyle', methods{i,5});
    end
    set(ax1,'YScale','log')
    xlabel(ax1,'$k/n$','Interpreter','latex')
    ylabel(ax1,'Sketching time (s)','Interpreter','latex')
    % if q == 1
    %     title(ax1,'Runtime','Interpreter','latex')
    % end
    xlim(ax1,[min(ks) max(ks)]/m); grid(ax1,'on')

    ax2 = axes('Parent',fig,'Position',[ax_left(2) bN ax_width hN]);
    hold(ax2,'on')
    for i = 1:nmeth
        plot_shaded(ks / m, smins(i,:,2,q), smins(i,:,1,q), smins(i,:,3,q), ...
                    methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                    'Marker', methods{i,4}, 'LineStyle', methods{i,5});
    end
    set(ax2,'YScale','log')
    xlabel(ax2,'$k/n$','Interpreter','latex')
    ylabel(ax2,'$\sigma_{\min}(\mbox{\boldmath $\Omega$}^\top \mbox{\boldmath $Q$})$', ...
           'Interpreter','latex')
    % if q == 1
    %     title(ax2,['Minimum singular value ' ...
    %                '$\sigma_{\rm min}(\mbox{\boldmath $\Omega$}^\top \mbox{\boldmath $Q$})$'], ...
    %           'Interpreter','latex')
    % end
    xlim(ax2,[min(ks) max(ks)]/m); ylim(ax2,[1e-3 1]); grid(ax2,'on')

    if q == 2
        % plot_shaded draws one Line (plus one shaded Patch) per method, so
        % the Lines in creation order line up with the method list.
        lines = flipud(findobj(ax2,'Type','Line'));

        % MATLAB fills a multi-column legend column by column, so this order
        % reads down each column in turn:
        %   Gaussian     CountSketch   Multisketch
        %   SRFT 1       SparseStack   Uniform
        %   SRFT 2       LESS          Leverage
        legend_order = [1 4 5 2 3 8 9 6 7];

        lgd = legend(lines(legend_order), methods(legend_order,1), ...
                     'Interpreter','latex', ...
                     'NumColumns', 3, 'Units','normalized');
        lgd.Position(1) = max(0.01, 0.5 - lgd.Position(3)/2);
        lgd.Position(2) = 0.03;
    end

    ansave(fig, fullfile(here,'..','..','figures', ...
                            ['sketching_comparison_' lower(basis_name{q})]));
end
