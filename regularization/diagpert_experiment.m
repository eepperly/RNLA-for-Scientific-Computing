here = fileparts(mfilename('fullpath'));

rng(0);
m = 5;
n = 2*m;
gam = 0.4;                  % size of the diagonal perturbation (gamma in the paper)
[Q,R] = qr(randn(n));
A = Q*blkdiag(zeros(m),eye(m))*Q';
D = diag(rand(n,1));
y = eig(A+gam*D)
plot(y, zeros(size(y)), 'xb', 'MarkerSize', 18, 'LineWidth', 2);

hold on

% Draw the horizontal black line (the real line)
plot([-0.1;1.4], zeros(2,1), 'k-', 'LineWidth', 1.2);
axis tight
axis equal
ylim([-0 0.1])

% Make it look clean and padded a bit
set(gca, 'XTick', [0,1], 'Box', 'off');
ax = gca;
ax.YAxis.Visible = 'off';   % hides y-axis only
ax.XAxis.Visible = 'on';    % keep x-axis visible (default)
ax.XAxis.FontSize = 16;    % works on modern MATLAB (graphics v2)
ax.TickLength = [0.02 0.02];  % longer ticks (first element used for 2-D)
ax.LineWidth  = 1.5;          % thicker axis lines & ticks
ax.TickDir    = 'both';        % (optional) ticks pointing outward

exportgraphics(ax, fullfile(here,'..','..','figures','diagpert.pdf'), ...
               'ContentType','vector');  % crops to axes box
