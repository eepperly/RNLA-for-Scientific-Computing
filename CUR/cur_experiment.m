
close all
doHMT = 0;
nRep = 20;  % Number of randomized repetitions for Random CUR, ARP, and ICUR
dosave = 0;

% Problem setup
n = 1000;
rs = 10:10:100;

A = diag(logspace(-40,0,n)); decay = 'exp';
%A = diag(logspace(-15,0,n)); decay = 'exp';
% A = diag((1:n).^(-2)); decay = 'poly';
m = n;

%[U,~] = qr(randn(n)); [Q,~] = qr(randn(n)); A = U*A*Q'; mat = 'Incoherent';
 [U,~] = qr(eye(n)+randn(n)*1e-5); [Q,~] = qr(eye(n)+randn(n)*1e-5); A = U*A*Q'; mat = 'Coherent';

[~,svdS,Vsvd] = svd(A);
sv = diag(svdS);

errs = [];
errHMT = [];

% Quantile matrices for randomized methods (10%, 50% median, 90%)
err5Quantiles = zeros(length(rs), 3); % Uniform Random CUR
err6Quantiles = zeros(length(rs), 3); % ARP / Randomized Osinsky

% -------------------------------------------------------------------------
% 1. Run ICUR across nRep repetitions
% -------------------------------------------------------------------------
block_size = rs(1);              % Block size per ICUR iteration
max_it     = length(rs);         % Total iterations to reach max rank
errICUR_Rep = zeros(nRep, max_it);

for rep = 1:nRep
    % Call icur (threshold = 0 ensures it runs all max_it iterations)
    [~, ~, ~, ~, ~, I_icur, J_icur] = icur(A, block_size, 0, max_it);
    
    % Compute actual ||A - CUR||_F error for each rank step
    for ir = 1:length(rs)
        cidx = J_icur(1 : ir * block_size);
        ridx = I_icur(1 : ir * block_size);
        cidx = cidx(cidx > 0);
        ridx = ridx(ridx > 0);
        
        C_icur = A(:, cidx);
        R_icur = A(ridx, :);
        W_icur = A(ridx, cidx);
        
        [UQ, UR] = qr(W_icur, 0);
        errICUR_Rep(rep, ir) = norm(A - (C_icur / UR) * (UQ' * R_icur), 'fro');
    end
end
errICURQuantiles = quantile(errICUR_Rep, [0.10 0.50 0.90], 1)';

% -------------------------------------------------------------------------
% 2. Sweep over ranks for deterministic & point-wise randomized methods
% -------------------------------------------------------------------------
for ir = 1:length(rs)

    r = rs(ir);

    % HMT/SVD right singular vectors
    if doHMT
        V = randn(r,n)*A;
        [V,~] = qr(V',0);

        errHMT = [errHMT; norm(A-(A*V)*V','fro')];
    else
        V = Vsvd(:,1:r);
    end

    % Osinsky
    %[ix,C,~] = Osinsky(A,V);
    [ix] = Osinsky(A,V);
    C = A(:,ix);

    [Q,~] = qr(C,0);

    %[ixx,R,~] = Osinsky(A',Q);
    [ixx] = Osinsky(A',Q);
    R = A(ixx,:);
%    R = R';

    [UQ,UR] = qr(A(ixx,ix),0);

    err1 = norm(A-(C/UR)*(UQ'*R), 'fro');

    % Osinsky with sketch norm
    [ixs,C,~] = Osinsky_sketchnorm(A,V);
    C = A(:,ixs);

    [Qsketch,~] = qr(C,0);

    [ixxs,R,~] = Osinsky_sketchnorm(A',Qsketch);
    R = R';

    [UQ,UR] = qr(A(ixxs,ixs),0);

    err3 = norm(A-(C/UR)*(UQ'*R), 'fro');

    % Uniformly random CUR: repeat nRep times
    err5Rep = zeros(nRep,1);
    for rep = 1:nRep
        ix = randperm(n,r);
        ixx = randperm(m,r);

        [UQ,UR] = qr(A(ixx,ix),0);

        err5Rep(rep) = norm(A-(A(:,ix)/UR)*(UQ'*A(ixx,:)), 'fro');
    end
    err5Quantiles(ir, :) = quantile(err5Rep, [0.10 0.50 0.90]);

    % Randomized ARP (err6): repeat nRep times
    err6Rep = zeros(nRep,1);
    for rep = 1:nRep
        ix = ARP(V);
        C = A(:,ix);

        [Q,~] = qr(C,0);

        ixx = ARP(Q);
        R = A(ixx,:);
        U = A(ixx,ix);

        [UQ,UR] = qr(U,0);

        err6Rep(rep) = norm(A-(A(:,ix)/UR)*(UQ'*A(ixx,:)), 'fro');
    end
    err6Quantiles(ir, :) = quantile(err6Rep, [0.10 0.50 0.90]);

    % Store errors (using medians)
    errnow = [ ...
        err1, ...
        norm(sv(r+1:end)), ...
        err5Quantiles(ir, 2), ...
        nan, ...
        err3, ...
        err6Quantiles(ir, 2)];

    errs = [errs; errnow];

end

%% Plot setup matching nystrom_experiment style

fig = figure('Units','normalized','Position',[0.15 0.15 0.60 0.55], ...
             'PaperPositionMode','auto');
ax = gca; hold on;

% Hex color palette matching nystrom_experiment
c_rand    = '#FE6100'; % Uniform Random CUR (Orange)
c_hmt     = '#785EF0'; % HMT (Purple)
c_osinsky = '#DC267F'; % Osinsky (Magenta)
c_icur    = '#009E73'; % ICUR (Teal/Green)
c_arp     = '#648FFF'; % ARP Quantile Shaded (Blue)
c_svd     = '#000000'; % SVD Optimal (Black)

% 1. Uniform Random CUR
plot_shaded(rs, err5Quantiles(:,2), err5Quantiles(:,1), err5Quantiles(:,3), ...
            c_rand, 'LineWidth', 1.5, 'LineStyle', '--', 'Marker', '.', 'MarkerSize', 12);

% 2. HMT (if enabled)
if doHMT
    plot(rs, errHMT, '.--', 'Color', c_hmt, 'LineWidth', 1.5, 'MarkerSize', 12);
end

% 3. Osinsky
plot(rs, errs(:,1), 'x-', 'Color', c_osinsky, 'LineWidth', 1.5, 'MarkerSize', 8);

% 4. ICUR (Median + 10%-90% shaded ribbon)
plot_shaded(rs, errICURQuantiles(:,2), errICURQuantiles(:,1), errICURQuantiles(:,3), ...
            c_icur, 'LineWidth', 1.5, 'LineStyle', '-.', 'Marker', 's', 'MarkerSize', 6);

% 5. ARP (Randomized CUR with 10%-90% shaded ribbon)
plot_shaded(rs, err6Quantiles(:,2), err6Quantiles(:,1), err6Quantiles(:,3), ...
            c_arp, 'LineWidth', 1.5, 'Marker', '.', 'MarkerSize', 12);

% 6. SVD Optimal Error
plot(rs, errs(:,2), 'k:', 'LineWidth', 1.5);

% Formatting and Axes
set(ax, 'YScale', 'log');
grid on;
xlim([min(rs) max(rs)]);

xlabel('Rank $r$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\|A - CUR\|_F$', 'Interpreter', 'latex', 'FontSize', 14);
title(['Matrix: ', mat], 'Interpreter', 'latex', 'FontSize', 14);

% Legend configuration
if doHMT
    labels = {'Random (median, 10--90\% range)', 'HMT', 'Osinsky', ...
              'IterativeCUR (median, 10--90\% range)', 'ARP (median, 10--90\% range)', 'SVD'};
else
    labels = {'Random (median, 10--90\% range)', 'Osinsky', ...
              'IterativeCUR (median, 10--90\% range)', 'ARP (median, 10--90\% range)', 'SVD'};
end
legend(labels, 'Interpreter', 'latex', 'FontSize', 13, 'Location', 'best');

set(gca,'fontsize',13)
shg

%% Optional export
if dosave
    name = ['CUROsinsky_example_n', num2str(n), mat, decay, 'doHMT', num2str(doHMT)];
    exportgraphics(gcf, [name, '.pdf'], 'ContentType', 'vector');    
end

%% Local functions

function h = plot_shaded(x, y_med, y_lo, y_hi, color, varargin)
% PLOT_SHADED Plots median curve with 10%-90% quantile shaded ribbon.
x = x(:)';
y_lo = y_lo(:)';
y_hi = y_hi(:)';

% Convert color (hex code, RGB triplet, or string) to valid RGB vector
if (ischar(color) || isstring(color)) && startsWith(color, '#')
    hex = char(color);
    c = sscanf(hex(2:end), '%2x')' / 255;
elseif ischar(color) || isstring(color)
    try
        c = validatecolor(color);
    catch
        c = color;
    end
else
    c = color;
end

% Draw 10%-90% shaded patch (hidden from legend using HandleVisibility)
fill([x, fliplr(x)], [y_lo, fliplr(y_hi)], c, ...
     'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Draw median line
h = plot(x, y_med, 'Color', color, varargin{:});
end