here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'..','utils'));

n = 1e3;
rho = 0.5;
d = n / rho;

trials = 100;
X = zeros(n,trials);

for trial = 1:trials
    X(:,trial) = svd(randn(d,n)/sqrt(d));
end

%% Plot

figure('Units', 'normalized', 'Position', [0, 0.4, 0.5, 0.3]);

hold on

bin_edges = linspace(0, 2.5, 101);  % Define bin edges

% Loop over each column of X and plot histogram with transparency
for i = 1:size(X, 2)
    % Plot histogram of the ith column with transparency, same color, no edge
    h = histogram(X(:,i), "Normalization", "pdf", "BinEdges", bin_edges);
    h.FaceColor = "#648FFF";  % Set same face color
    h.EdgeColor = 'none';  % Remove black outlines
    h.FaceAlpha = 0.02;  % Set transparency
end

smin = 1 - sqrt(rho);
smax = 1 + sqrt(rho);
xx = linspace(smin,smax,200);
plot(xx,sqrt((smax^2-xx.^2).*(xx.^2-smin^2)) ./ (pi*rho*xx),"k--")

axis([0 2 0 1])

% Add labels and title
xlabel('$\sigma_i(\mbox{\boldmath $\Omega$}^\top\mbox{\boldmath $Q$})$')
ylabel('Probability density')

ansave(gcf, fullfile(here,'..','..','figures','mp'));
