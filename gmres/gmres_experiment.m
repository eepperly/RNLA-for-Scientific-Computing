%% GMRES variants on the advection-diffusion problem
%
% The operator is the one from the sketching experiment: the standard
% five-point Laplacian plus centered advection on an n1-by-n1 interior grid of
% the unit square with homogeneous Dirichlet boundary conditions, so
% N = n1^2 = 1.6*10^5.  The grid Peclet number is small, so the problem stays
% diffusion dominated and centered differences need no upwinding.  The
% right-hand side is f(x,y) = sin(x)*cos(y) on the grid.
%
% Five solvers:
%
%   GMRES              full Arnoldi.  Optimal residual over the Krylov space,
%                      but O(N*j) orthogonalization at step j and the whole
%                      basis in memory.
%   GMRES(restart)     the same, restarted every `restart` steps.  Bounded
%                      cost and storage, at the price of discarding the space.
%   sGMRES (p=8)       truncated-Arnoldi basis, orthogonalized against only the
%                      previous p vectors, with the least-squares problem
%                      solved in the sketched space (Nakatsukasa and Tropp
%                      2024).
%   sGMRES (p=2)       the same with a shallower truncation.  Nakatsukasa and
%                      Tropp report p = 2 or 4 usually giving a basis of
%                      moderate condition number; the two lines show what the
%                      extra orthogonalization buys.
%   RGS-GMRES          randomized Gram-Schmidt basis, orthonormal in the
%                      sketched inner product, one pass over the basis per
%                      iteration (Balabanov and Grigori 2022).
%
% Every solver is timed the same way -- basis build, small least-squares
% solve, and forming x -- with the residual evaluation excluded as
% measurement.  Both panels are against the iteration count: the left shows
% what an iteration buys in accuracy, the right what it costs in time.  Read
% together they are the trade: full GMRES gets the most out of each iteration
% and pays the most for it, and the other four give some of that back.
%
% WARNING: each solver stores a basis of m columns of length N, i.e. 8*N*m
% bytes; sGMRES additionally holds the sketched reduced matrix.  With the
% defaults below that is roughly 1.3 GB per solver, one solver at a time.  Reduce n1 or m if that is too much.

n1      = 400;         % interior grid points per dimension; N = n1^2
Pe      = 0.1;         % grid Peclet number
m       = 1000;        % iterations
restart = 50;          % cycle length for restarted GMRES
s_orth  = [8 2];       % steps of partial orthogonalization (s in the paper),
                       % one sGMRES line per value
trials  = 100;         % repetitions; the randomized methods vary between runs

N = n1^2;
h = 1/(n1+1);
c = 2*Pe/h;

here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here,'..','utils'), fullfile(here,'..','sketching'));

%% Operator and right-hand side

e  = ones(n1,1);
T  = spdiags([-e 2*e -e], -1:1, n1, n1) / h^2;   % 1D Dirichlet Laplacian
D  = spdiags([-e 0*e e], -1:1, n1, n1) / (2*h);  % 1D centered difference
I1 = speye(n1);
A  = kron(I1,T) + kron(T,I1) + c*(kron(I1,D) + kron(D,I1));

xg = (1:n1)' * h;
[X,Y] = ndgrid(xg,xg);
b = sin(X) .* cos(Y);
b = b(:);

chk = 50:50:m;                                   % iteration counts to report
nchk = numel(chk);

%          name                solver                                          color      mark style  hasT
methods = { ...
    'GMRES',            @() gmres_standard(A,b,m,chk),                  '#648FFF','o','-',  false; ...
    'GMRES(50)',        @() gmres_restarted(A,b,m,chk,restart),         '#DC267F','s','--', false; ...
    'sGMRES ($p=8$)',   @() sgmres(A,b,m,chk,s_orth(1)),                '#FE6100','d','-.', true;  ...
    'sGMRES ($p=2$)',   @() sgmres(A,b,m,chk,s_orth(2)),                '#785EF0','v',':',  true;  ...
    'RGS-GMRES',        @() rgs_gmres(A,b,m,chk),                       '#009E73','^','-',  false};
nmeth = size(methods,1);

%% Run

tt = nan(nmeth, nchk, 3);
rr = nan(nmeth, nchk, 3);
cc = nan(nmeth, nchk, 3);       % condest(T); only sGMRES has a whitening factor

wall = tic;
for i = 1:nmeth
    Tt = nan(trials, nchk);
    Rr = nan(trials, nchk);
    Ct = nan(trials, nchk);
    tw = tic;
    for t = 1:trials
        if methods{i,6}
            [Tt(t,:), Rr(t,:), Ct(t,:)] = methods{i,2}();
        else
            [Tt(t,:), Rr(t,:)] = methods{i,2}();
        end
    end
    wallmeth = toc(tw);
    tt(i,:,:) = permute(quantile(Tt, [0.10 0.50 0.90], 1), [3 2 1]);
    rr(i,:,:) = permute(quantile(Rr, [0.10 0.50 0.90], 1), [3 2 1]);
    if methods{i,6}
        cc(i,:,:) = permute(quantile(Ct, [0.10 0.50 0.90], 1), [3 2 1]);
    end
    % Two different quantities, so both are named.  The solver time is the
    % cost of building to the last checkpoint and solving once -- what the plot
    % shows.  The wall time additionally includes the checkpoint solve and
    % residual evaluation repeated at all %d checkpoints, which is measurement
    % scaffolding rather than solver cost.
    fprintf('  %-14s final relres %.3e | solver %6.1f s | wall %6.1f s (%.0f%% harness) | total %.1f s\n', ...
            methods{i,1}, rr(i,end,2), tt(i,end,2), wallmeth/trials, ...
            100*max(0, wallmeth/trials - tt(i,end,2))/(wallmeth/trials), toc(wall));
end
fprintf('Experiment finished in %.1f s\n', toc(wall));

%% Plot: accuracy and cost, both against the iteration count

% Geometry in pixels, so the export has a deterministic size.  These are the
% same numbers as the other two-panel experiments, so the figures come out the
% same size on the page.
fig_w     = 1750;
fig_h     = 620;
legend_h  = 110;            % px, just enough for the legend under the x labels
ax_left   = [150 1005];
ax_width  = 630;
ax_bottom = 140;
ax_height = 375;

fig = figure('Units','pixels','Position',[100 100 fig_w fig_h+legend_h], ...
             'PaperPositionMode','auto');

ax1 = axes('Parent',fig,'Units','pixels', ...
           'Position',[ax_left(1) ax_bottom+legend_h ax_width ax_height]);
ax1.Toolbar.Visible = 'off';            % keeps the toolbar out of the export
hold(ax1,'on')
for i = 1:nmeth
    plot_shaded(chk, rr(i,:,2), rr(i,:,1), rr(i,:,3), ...
                methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
set(ax1,'YScale','log')
xlabel(ax1,'Iteration','Interpreter','latex')
ylabel(ax1,['$\|\mbox{\boldmath $A$}\mbox{\boldmath $x$}-\mbox{\boldmath $b$}\|' ...
            '/\|\mbox{\boldmath $b$}\|$'],'Interpreter','latex')
title(ax1,'Relative residual vs iteration','Interpreter','latex')
xlim(ax1,[min(chk) max(chk)]); grid(ax1,'on')

ax2 = axes('Parent',fig,'Units','pixels', ...
           'Position',[ax_left(2) ax_bottom+legend_h ax_width ax_height]);
ax2.Toolbar.Visible = 'off';
hold(ax2,'on')
for i = 1:nmeth
    plot_shaded(chk, tt(i,:,2), tt(i,:,1), tt(i,:,3), ...
                methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
set(ax2,'YScale','log')
xlabel(ax2,'Iteration','Interpreter','latex')
ylabel(ax2,'Time (s)','Interpreter','latex')
title(ax2,'Time vs iteration','Interpreter','latex')
xlim(ax2,[min(chk) max(chk)]); grid(ax2,'on')

lines = flipud(findobj(ax2,'Type','Line'));
lgd = legend(lines, methods(:,1), 'Interpreter','latex', ...
             'NumColumns', 5, 'Units','pixels');
lgd.Position(1) = max(0, (fig_w - lgd.Position(3))/2);
lgd.Position(2) = 25;

ansave(fig, fullfile(here,'..','..','figures','gmres_advdiff'));

%% Second figure: the whitening condition number, sGMRES only

% Same width as the figure above so both scale identically when included at a
% common width; the single panel is centred rather than stretched.
c_left   = 425;
c_width  = 900;

hasT = find(cell2mat(methods(:,6)))';

figc = figure('Units','pixels','Position',[100 100 fig_w fig_h+legend_h], ...
              'PaperPositionMode','auto');
axc = axes('Parent',figc,'Units','pixels', ...
           'Position',[c_left ax_bottom+legend_h c_width ax_height]);
axc.Toolbar.Visible = 'off';
hold(axc,'on')
for i = hasT
    plot_shaded(chk, cc(i,:,2), cc(i,:,1), cc(i,:,3), ...
                methods{i,3}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
                'Marker', methods{i,4}, 'LineStyle', methods{i,5});
end
% Drawn with plot, not yline: yline makes a ConstantLine, which findobj
% below would not return, leaving the legend one handle short of its labels.
% sGMRES is robust while cond(A*B) stays under about 1/u (section 3.4); u is
% the unit roundoff, ~2e-16 by their footnote, so 1/eps is the right line.
plot(axc, chk, (1/eps)*ones(size(chk)), 'k:', 'LineWidth', 1.2);
set(axc,'YScale','log')
xlabel(axc,'Iteration','Interpreter','latex')
ylabel(axc,'$\mathrm{condest}(\mbox{\boldmath $T$})$','Interpreter','latex')
title(axc,'Whitening condition number vs iteration','Interpreter','latex')
xlim(axc,[min(chk) max(chk)]); grid(axc,'on')

linesc = flipud(findobj(axc,'Type','Line'));
lgdc = legend(linesc, [methods(hasT,1); {'$1/u$'}], 'Interpreter','latex', ...
              'NumColumns', 3, 'Units','pixels');
lgdc.Position(1) = max(0, (fig_w - lgdc.Position(3))/2);
lgdc.Position(2) = 25;

ansave(figc, fullfile(here,'..','..','figures','gmres_advdiff_condT'));
