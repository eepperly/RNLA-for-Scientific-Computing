%% Low-rank approximation of synthetic matrices: four methods, two spectra
%
% A = V*diag(d)*V' is n-by-n with n = 5000, V a fixed random orthogonal matrix
% shared by both cases so the spectrum is the only thing that differs between
% them.  Two spectra, one per figure:
%
%   Step (top)         d = [1 (20 times), 1e-4 (the remaining n-20)].  A clean
%                      rank-20 signal over a flat floor, so the best rank-k
%                      error is 1 for k < 20 and exactly 1e-4 for k >= 20.
%   Algebraic (bottom) d_j = j^{-2}.  No gap, and the best rank-k error decays
%                      smoothly as (k+1)^{-2}.
%
% Four approximations, all of rank k, each returned in factored form U*V:
%
%   Single-pass Nystrom   Ahat = (AX)(X'AX)^+ (AX)',  X sparse.  k matvecs.
%   Projection            Y = AX, [Q,~] = qr(Y,'econ'), Ahat = Q(Q'A).  2k.
%   Two-pass Nystrom      Y = AX orthogonalized by QR, then Nystrom with that
%                         orthonormal test matrix.  2k.
%   Generalized Nystrom   Ahat = (AX)(Y'AX)^+ (Y'A), X sparse n-by-k and Y
%                         sparse n-by-2k.  3k.
%
% Generalized Nystrom follows the survey: the core
% pseudoinverse is applied through a thin QR, Y'AX = QR, as R^{-1}Q', and the
% result is kept as an outer product U*V.  The order of multiplication matters
% and is not rearranged.
%
% Nothing dense of size n-by-n is formed beyond A itself.  Approximations stay
% factored, and the error ||A - U*V||_2 is computed by svds on the residual
% applied implicitly, rather than forming the residual and taking its SVD.
% This keeps the per-trial cost at O(n^2) so n can be raised without
% rewriting the harness.
%
% Left panels plot relative spectral-norm error against matrix-vector products,
% right panels against wall-clock time.

n      = 5e3;          % matrix dimension
ks     = 15:15:300;    % approximation ranks to sweep
trials = 30;           % repetitions, for the median and 10/90 bands

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'), fullfile(here,'..','sketching'));

% Named Qeig, not V: the sweep below unpacks [U,V] = approximate(...), which
% would otherwise clobber the eigenbasis after the first iteration.
[Qeig, ~] = qr(randn(n));               % shared random orthogonal eigenbasis

mats = { ...
    'Step spectrum',    [ones(20,1); 1e-4*ones(n-20,1)], 'step'; ...
    'Algebraic decay',  (1:n)'.^(-2),                    'algebraic'};

%          name                     matvecs per rank k   color      mark  style
methods = { ...
    'Single-pass Nystr\"om',   1, '#648FFF', 'o', '-'; ...
    'Projection',            2, '#DC267F', 's', '--'; ...
    'Two-pass Nystr\"om',      2, '#FE6100', 'd', '-.'; ...
    'Generalized Nystr\"om',   3, '#009E73', '^', '-'};
nmeth = size(methods,1);
nk    = numel(ks);

err  = zeros(nmeth, nk, 3, 2);
tim  = zeros(nmeth, nk, 3, 2);
mvs  = zeros(nmeth, nk);
for i = 1:nmeth
    mvs(i,:) = methods{i,2} * ks;
end

%% Sweep

wall = tic;
for q = 1:2
    d     = mats{q,2};
    A     = (Qeig .* d') * Qeig';
    A     = (A + A')/2;                 % kill the asymmetry rounding leaves
    normA = normest(A, 1e-10);
    fprintf('%s: ||A|| = %.4e, best rank-%d error %.3e\n', ...
            mats{q,1}, normA, max(ks), d(max(ks)+1)/d(1));

    for i = 1:nmeth
        for j = 1:nk
            k  = ks(j);
            te = zeros(trials,1);
            tt = zeros(trials,1);
            for t = 1:trials
                tic;
                [U, V] = approximate(A, n, k, i);
                tt(t) = toc;

                % ||A - U*V||_2 with the residual applied implicitly.  A is
                % symmetric, but U*V need not be, so this is svds not eigs.
                res   = @(x,tflag) residual_apply(A, U, V, x, tflag);
                te(t) = svds(res, [n n], 1, 'largest', 'Tolerance', 1e-6) / normA;
            end
            err(i,j,:,q) = quantile(te, [0.10 0.50 0.90]);
            tim(i,j,:,q) = quantile(tt, [0.10 0.50 0.90]);
        end
        fprintf('  %-22s done, %.1f s elapsed\n', methods{i,1}, toc(wall));
    end
    clear A
end
fprintf('Sweep finished in %.1f s\n', toc(wall));

%% Plot: one figure per kernel, matvecs on the left, time on the right

% Geometry in pixels rather than normalized units.  A normalized figure size is
% resolved against the screen when the figure is saved, and the two exports came
% out 1772 and 1669 pixels wide despite both asking for the same fraction of the
% screen.  Dropped into LaTeX at a common \includegraphics width those scale by
% different factors, so the panels cannot line up.  Fixing the pixel size makes
% both exports exactly the same width, which puts the axes at the same place on
% the page.  The figure carrying the legend is taller by legend_h; because these
% are absolute units, its axes keep the same size and simply sit higher.
%
% fig_w must fit on the screen, or MATLAB clamps the figure and the widths
% diverge again.
fig_w     = 1750;           % px, shared by both figures
fig_h     = 620;            % px, height without the legend strip
legend_h  = 190;            % px, extra height on the figure carrying the legend
ax_left   = [150 1005];     % px, left edge of each axes
ax_width  = 630;            % px
ax_bottom = 140;            % px, leaving room for ticks and the x label
ax_height = 375;            % px, leaving room for the title above

for q = 1:2
    extra = (q == 2) * legend_h;
    H     = fig_h + extra;

    fig = figure('Units','pixels','Position',[100 100 fig_w H], ...
                 'PaperPositionMode','auto');

    ax1 = axes('Parent',fig,'Units','pixels', ...
               'Position',[ax_left(1) ax_bottom+extra ax_width ax_height]);
    hold(ax1,'on')
    for i = 1:nmeth
        plot_shaded(mvs(i,:), err(i,:,2,q), err(i,:,1,q), err(i,:,3,q), ...
                    methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                    'Marker', methods{i,4}, 'LineStyle', methods{i,5});
    end
    set(ax1,'XScale','log','YScale','log')
    xlabel(ax1,'Matrix--vector products','Interpreter','latex')
    ylabel(ax1,['$\|\mbox{\boldmath $A$}-\widehat{\mbox{\boldmath $A$}}\|' ...
                '/\|\mbox{\boldmath $A$}\|$'],'Interpreter','latex')
    if q == 1
        title(ax1,'Error vs matrix--vector products','Interpreter','latex')
    end
    grid(ax1,'on')

    ax2 = axes('Parent',fig,'Units','pixels', ...
               'Position',[ax_left(2) ax_bottom+extra ax_width ax_height]);
    hold(ax2,'on')
    for i = 1:nmeth
        plot_shaded(tim(i,:,2,q), err(i,:,2,q), err(i,:,1,q), err(i,:,3,q), ...
                    methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                    'Marker', methods{i,4}, 'LineStyle', methods{i,5});
    end
    set(ax2,'XScale','log','YScale','log')
    xlabel(ax2,'Time (s)','Interpreter','latex')
    ylabel(ax2,['$\|\mbox{\boldmath $A$}-\widehat{\mbox{\boldmath $A$}}\|' ...
                '/\|\mbox{\boldmath $A$}\|$'],'Interpreter','latex')
    if q == 1
        title(ax2,'Error vs time','Interpreter','latex')
    end
    grid(ax2,'on')

    if q == 2
        lines = flipud(findobj(ax2,'Type','Line'));
        lgd = legend(lines, methods(:,1), 'Interpreter','latex', ...
                     'NumColumns', 4, 'Units','pixels');
        lgd.Position(1) = max(0, (fig_w - lgd.Position(3))/2);
        lgd.Position(2) = 40;
    end

    ansave(fig, fullfile(here,'..','..','figures', ...
                            ['synthetic_lowrank_' mats{q,3}]));
end

%% Local functions

function [U, V] = approximate(A, n, k, which)
%APPROXIMATE  Rank-k approximation of A by the requested method, as Ahat = U*V.

switch which

    case 1  % Single-pass Nystrom, sparse test matrix
        [~,~,Omt] = SparseStack(zeros(size(A,1),0),[],k,4);
        [W, Lam] = nystrom_factor(A, Omt');     % shifted, so Cholesky is safe
        U = W .* Lam';
        V = W';

    case 2  % Projection: orthonormalize the sketch, then a second pass
        Y = A * randn(n,k);
        [U, ~] = qr(Y, 'econ');
        V = U' * A;

    case 3  % Two-pass Nystrom, orthogonalized after the first round
        Y = A * randn(n,k);             % k matvecs
        [Q, ~] = qr(Y, 'econ');         % QR keeps the test matrix well
        [W, Lam] = nystrom_factor(A, Q);% conditioned for the second round
        U = W .* Lam';
        V = W';

    case 4  % Generalized Nystrom, outer product form
        [~,~,Omt] = SparseStack(zeros(size(A,1),0),[],k,4);
        [~,~,Psit] = SparseStack(zeros(size(A,1),0),[],2*k,4);
        AOm = (Omt * A)';
        PsitA = Psit*A;                         
        [Q, R] = qr(Psit * AOm, 0);              
        U = AOm / R;                             
        V = Q' * PsitA;                            
end

end

function y = residual_apply(A, U, V, x, tflag)
%RESIDUAL_APPLY  Apply A - U*V or its transpose, without forming either.
if strcmp(tflag, 'notransp')
    y = A*x - U*(V*x);
else
    y = A*x - V'*(U'*x);                % A is symmetric
end
end