%% Reproduce Table 1 of "Randomly pivoted Cholesky" (arXiv:2207.06503)
%
% Chen, Epperly, Tropp and Webber, "Randomly pivoted Cholesky: Practical
% approximation of a kernel matrix with few entry evaluations", CPAM 2025.
% Table 1 compares column Nystrom methods by the relative trace error
%
%       ( tr(A) - tr(Ahat) ) / tr(A)
%
% of a rank-1000 approximation of a Gaussian kernel matrix, on twenty datasets
% from LIBSVM, OpenML and UCI.  This script follows the reference script
% (experiments/many_matrices.py in github.com/eepperly/Randomly-Pivoted-Cholesky)
% step for step:
%
%   data       the training split Xtr of each dataset, made dense
%   scaling    sklearn's StandardScaler, fit on all of Xtr: subtract the mean,
%              divide by the population standard deviation, leaving
%              (numerically) constant features unscaled
%   points     the FIRST min(n, 10^4) rows of the scaled training set -- not a
%              random subsample, so the train/test split decides which points
%              enter the kernel
%   kernel     Gaussian, exp(-||x-y||^2 / (2 sigma^2)), sigma = sqrt(d) with d
%              the number of columns of Xtr
%   rank       k = 1000;  trials = 10;  median of the relative trace error
%   Opt        sum of the eigenvalues beyond the 1000th, over the trace
%
% Methods.  Block RPCholesky is removed from the comparison and nuclear
% selection added:
%
%   Uniform     k indices uniformly without replacement
%   RLS         recursive ridge leverage score sampling (RLS_NYSTROM).  The
%               reference calls recursive_rls_acc, but the recursiveNystrom it
%               wraps never reads its accelerated_flag argument, so that is the
%               same algorithm as plain recursive RLS, which is what RLS_NYSTROM
%               ports.
%   Greedy      greedy pivoted Cholesky (PIVOTED_CHOLESKY, 'greedy')
%   RPChol      simple RPCholesky (PIVOTED_CHOLESKY, 'rp'), as in the reference;
%               accelerated RPCholesky has the same output distribution
%   Nuclear     nuclear score selection (NUCLEAR_CHOLESKY), new here
%
% Greedy and nuclear selection are deterministic, so they are run once; the
% median of ten identical values is that value.
%
% The trace of the approximation is computed as the reference computes it:
% from the Cholesky factor, ||F||_F^2, for the pivoted Cholesky methods, and
% for the sampling methods from the rows A(S,:) through a Cholesky
% factorization of A(S,S) + tr(A(S,S))*eps*I (EXTENSION_TRACE).  On the most
% compressible datasets the errors reach 1e-11, where these choices, and
% roundoff generally, show in the digits.
%
% RUNNING IN PIECES.  The full table takes about an hour, mostly nuclear
% selection (~2 min per dataset) and the eigendecompositions for Opt.  Each
% dataset's result is saved to data/rpcholesky_table/<name>.mat as soon
% as it finishes, and a rerun skips datasets already saved, so the work
% survives interruption.  Set only (indices into the dataset list) before
% running to process a subset; the table is assembled and written once every
% dataset has a saved result.  Delete the saved files to recompute.
%
% DATA.  Set datadir before running to choose the input directory; by default
% it is the output directory of DOWNLOAD_DATA.  To reproduce the published
% numbers on the identical matrices, point it at the output of the authors'
% Python download_data.py instead: for the datasets split at random, the
% MATLAB loader's split (MATLAB's generator) differs from sklearn's, so its
% first 10^4 training points -- though drawn from identical data -- are a
% different, equally valid subsample.

here = fileparts(mfilename('fullpath'));
addpath(here);
if ~exist('datadir', 'var') || isempty(datadir)
    datadir = fullfile(here, '..', 'data', 'rpcholesky', 'preprocessed');
end

N      = 1e4;
k      = 1000;
trials = 10;

%       file                   label in the paper
data = {'sensit_vehicle',      'sensit\_vehicle'; ...
        'yolanda',             'yolanda'; ...
        'YearPredictionMSD',   'YearPredictionMSD'; ...
        'w8a',                 'w8a'; ...
        'MNIST',               'MNIST'; ...
        'jannis',              'jannis'; ...
        'HIGGS',               'HIGGS'; ...
        'connect-4',           'connect\_4'; ...
        'volkert',             'volkert'; ...
        'creditcard',          'creditcard'; ...
        'Medical-Appointment', 'Medical\_Appointment'; ...
        'sensorless',          'sensorless'; ...
        'ACSIncome',           'ACSIncome'; ...
        'Airlines_DepDelay_1M','Airlines\_DepDelay\_1M'; ...
        'covtype.binary',      'covtype\_binary'; ...
        'diamonds',            'diamonds'; ...
        'hls4ml_lhc_jets_hlf', 'hls4ml\_lhc\_jets\_hlf'; ...
        'ijcnn1',              'ijcnn1'; ...
        'cod-rna',             'cod\_rna'; ...
        'COMET_MC_SAMPLE',     'COMET\_MC\_SAMPLE'};

% Published values (arXiv v6, Table 1) for the columns kept here:
%          Unif     RLS      Greedy   RPChol   Opt
paper = [ 1.57e-1  1.40e-1  2.07e-1  1.37e-1  8.77e-2
          1.46e-1  1.39e-1  2.08e-1  1.36e-1  8.41e-2
          1.30e-1  1.20e-1  1.73e-1  1.16e-1  6.79e-2
          1.42e-1  1.17e-1  1.91e-1  1.05e-1  6.09e-2
          1.21e-1  1.10e-1  1.67e-1  1.06e-1  5.83e-2
          1.11e-1  1.10e-1  1.28e-1  1.09e-1  5.45e-2
          6.73e-2  6.31e-2  8.51e-2  6.07e-2  2.96e-2
          5.81e-2  5.07e-2  6.48e-2  4.81e-2  2.25e-2
          5.35e-2  4.42e-2  5.92e-2  4.17e-2  1.99e-2
          4.83e-2  3.71e-2  5.77e-2  3.01e-2  1.31e-2
          1.74e-2  1.43e-2  1.92e-2  1.29e-2  4.59e-3
          1.20e-2  7.78e-3  8.70e-3  5.80e-3  2.11e-3
          9.93e-3  5.55e-3  8.35e-3  4.02e-3  1.27e-3
          4.19e-3  2.37e-3  2.64e-3  1.78e-3  5.08e-4
          9.10e-3  2.12e-3  1.41e-3  1.04e-3  2.97e-4
          1.31e-3  2.40e-4  1.12e-4  5.85e-5  1.30e-5
          3.78e-4  7.30e-5  6.68e-5  4.38e-5  1.04e-5
          3.91e-5  3.09e-5  2.86e-5  2.13e-5  4.67e-6
          6.00e-4  1.36e-5  9.19e-6  5.05e-6  9.86e-7
          3.65e-3  2.44e-7  1.2e-10  4.3e-11  3.5e-12];

cols = {'Unif', 'RLS', 'Greedy', 'RPChol', 'Nuclear', 'Opt'};
nd   = size(data, 1);
if ~exist('only', 'var') || isempty(only)
    only = 1:nd;
end

ckdir = fullfile(here, '..', 'data', 'rpcholesky_table');
if ~isfolder(ckdir), mkdir(ckdir); end
ckfile = @(q) fullfile(ckdir, [data{q,1} '.mat']);

wall = tic;
for q = only
    if isfile(ckfile(q))
        fprintf('%-22s already done, skipping\n', data{q,1});
        continue
    end
    rng(q);                                     % per dataset, so pieces reproduce
    %% Data, exactly as the reference prepares it
    S = load(fullfile(datadir, [data{q,1} '.mat']), 'Xtr');
    X = full(S.Xtr);
    X = standard_scaler(X);
    d = size(X, 2);
    X = X(1:min(N, size(X,1)), :);
    n = size(X, 1);

    A = gauss_kernel(X, X, sqrt(d));
    A = (A + A') / 2;
    A(1:n+1:end) = 1;                           % exp(0), as the reference's diag()
    trA = n;
    clear S X

    %% Methods
    e = struct('Unif', zeros(trials,1), 'RLS', zeros(trials,1), ...
               'RPChol', zeros(trials,1));
    K = MatrixOracle(A);
    for t = 1:trials
        Su = randsample(n, k, false);
        e.Unif(t) = (trA - extension_trace(A(Su,:), Su)) / trA;

        Sr = rls_nystrom(K, k);
        e.RLS(t) = (trA - extension_trace(A(Sr,:), Sr)) / trA;

        [~, F] = pivoted_cholesky(A, k, 'rp');
        e.RPChol(t) = (trA - norm(F, 'fro')^2) / trA;
    end
    [~, F] = pivoted_cholesky(A, k, 'greedy');
    e.Greedy = (trA - norm(F, 'fro')^2) / trA;
    [~, F] = nuclear_cholesky(A, k);
    e.Nuclear = (trA - norm(F, 'fro')^2) / trA;
    e.NuclearPivots = size(F, 2);               % < k if A ran out of numerical rank
    if e.NuclearPivots < k
        fprintf('  nuclear selection stopped at %d pivots (numerical rank)\n', e.NuclearPivots);
    end
    clear F K

    ev = sort(eig(A), 'descend');
    e.Opt = sum(ev(k+1:end)) / sum(ev);
    clear A

    row = zeros(1, numel(cols));
    for c = 1:numel(cols)
        row(c) = median(e.(cols{c}));
    end
    save(ckfile(q), 'row', 'e', 'cols', 'datadir', 'N', 'k', 'trials');
    fprintf('%-22s %s   (%.0f s elapsed)\n', data{q,1}, ...
            sprintf('%10.3g', row), toc(wall));
end

%% Output, once every dataset has a saved result

done = arrayfun(@(q) isfile(ckfile(q)), 1:nd);
if ~all(done)
    fprintf('\n%d of %d datasets done; still to run: %s\n', nnz(done), nd, ...
            mat2str(find(~done)));
    return
end

res = nan(nd, numel(cols));
raw = cell(nd, 1);
src = cell(nd, 1);
for q = 1:nd
    C = load(ckfile(q));
    res(q,:) = C.row;
    raw{q}   = C.e;
    src{q}   = C.datadir;
end
% Sanity check.  No approximation of rank <= k can have a smaller trace error
% than the optimal rank-k approximation, so an entry below Opt (by more than
% roundoff, N*eps in these relative units) is impossible and signals a
% numerical breakdown.  Such entries are reported as '---' rather than as a
% number.  This catches nuclear selection on COMET_MC_SAMPLE: that matrix is
% low-rank to working precision, nuclear scores tie once the residual is
% nearly low-rank, and roundoff then picks a pivot whose residual diagonal is
% ~1e-13 and known to only a few digits; normalizing by its square root
% destroys the factor.  (Elsewhere the nuclear factor was checked against the
% trace recomputed from its pivot set, agreeing to 1e-7 or better.)
optc = find(strcmp(cols, 'Opt'));
bad  = res < res(:, optc) - N*eps;
bad(:, optc) = false;

save(fullfile(here, '..', 'data', 'rpcholesky_table_results.mat'), ...
     'res', 'bad', 'raw', 'cols', 'data', 'paper', 'src', 'N', 'k', 'trials');

fprintf('\nRelative trace error, rank %d, median of %d trials\n', k, trials);
fprintf('%-22s', 'Dataset'); fprintf('%11s', cols{:}); fprintf('\n');
for q = 1:nd
    fprintf('%-22s', data{q,1});
    for c = 1:numel(cols)
        if bad(q,c), fprintf('%11s', '---'); else, fprintf('%11.2e', res(q,c)); end
    end
    fprintf('\n');
end
if any(bad(:))
    [qq, cc] = find(bad);
    for m = 1:numel(qq)
        fprintf('  ---: %s on %s computed %.3g, below Opt = %.3g; numerical breakdown\n', ...
                cols{cc(m)}, data{qq(m),1}, res(qq(m),cc(m)), res(qq(m),optc));
    end
end

shared = [1 2 3 4 6];                           % columns also in the paper
fprintf('\nRatio to the published value (1 = reproduced)\n');
fprintf('%-22s', 'Dataset'); fprintf('%11s', cols{shared}); fprintf('\n');
for q = 1:nd
    fprintf('%-22s', data{q,1}); fprintf('%11.3g', res(q,shared) ./ paper(q,:)); fprintf('\n');
end

% LaTeX table, in the paper's layout with nuclear selection in place of
% block RPCholesky.
texfile = fullfile(here, '..', '..', 'figures', 'rpcholesky_table.tex');
fid = fopen(texfile, 'w');
fprintf(fid, '%% Generated by column-nystrom/nystrom_table_experiment.m\n');
fprintf(fid, '%% ---: numerical breakdown (computed error below the optimal rank-k error)\n');
fprintf(fid, '\\begin{tabular}{l%s}\n\\toprule\n', repmat('c', 1, numel(cols)));
fprintf(fid, 'Dataset & %s \\\\\n\\midrule\n', strjoin(cols, ' & '));
for q = 1:nd
    cc = sci(res(q,:));
    cc(bad(q,:)) = {'---'};
    fprintf(fid, '%s', data{q,2});
    fprintf(fid, ' & %s', cc{:});
    fprintf(fid, ' \\\\\n');
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
fclose(fid);
fprintf('\nWrote %s\n', texfile);

%% Local functions

function X = standard_scaler(X)
%STANDARD_SCALER  sklearn.preprocessing.StandardScaler().fit_transform(X).
%   Population (ddof = 0) standard deviation, and features that are constant
%   to working precision -- var <= n*eps*var + (n*mean*eps)^2, sklearn's
%   _is_constant_feature -- are centred but not scaled.
n  = size(X, 1);
mu = mean(X, 1);
X  = X - mu;
v  = mean(X.^2, 1);
const = v <= n*eps*v + (n*mu*eps).^2;
sd = sqrt(v);
sd(const) = 1;
X = X ./ sd;
end

function c = sci(v)
%SCI  Entries in the paper's x.xxe-y format.
c = cell(1, numel(v));
for j = 1:numel(v)
    c{j} = regexprep(sprintf('%.2e', v(j)), 'e([+-])0?(\d+)', 'e$1$2');
end
c = c(:)';
end
