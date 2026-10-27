function [S, C, lev, lam, perm] = rls_nystrom(K, s)
%RLS_NYSTROM  Column Nystrom landmarks by recursive ridge leverage score sampling.
%   [S,C] = RLS_NYSTROM(K,s) selects s distinct landmarks by the recursive
%   ridge leverage score sampler of Musco and Musco, "Recursive sampling for the
%   Nystrom method" (NeurIPS 2017), a linear-time RLS sampler.  Returns the
%   indices S and the columns C = A(:,S).  K is a KERNELORACLE.
%
%   [S,C,lev,lam,perm] = RLS_NYSTROM(...) also returns the top-level leverage
%   score estimates lev, the top-level ridge parameter lam, and the internal
%   permutation perm; lev(i) is the estimate for point perm(i).
%
%   ALGORITHM.  The points are randomly permuted and nested uniform subsets of
%   halving size are formed.  Starting from the smallest, each level estimates
%   the lambda-ridge leverage scores of the next larger subset using a Nystrom
%   approximation built from the landmarks of the level below, and resamples
%   landmarks by those estimates.  lambda is set at each level from the
%   spectrum, as the tail mass beyond the top k = ceil(s/(4 log s)) eigenvalues
%   divided by k.  At the top level, s landmarks are drawn without replacement
%   with probability proportional to the estimated ridge leverage scores.
%
%   This is a line-by-line port of the non-accelerated reference implementation
%   (recursiveNystrom, Python port of the authors' MATLAB code, as used in the
%   RPCholesky papers).  Two numerical guards are added, neither of which
%   triggers in exact arithmetic: the regularized solve uses backslash rather
%   than an explicit inverse, and a nonpositive lambda (possible only through
%   roundoff, as the tail mass is nonnegative) falls back to the reference's
%   own small-sample value 1e-5.
%
%   Entry accesses: n for the diagonal, |current subset| * |landmarks| at each
%   level, and n*s for the final columns.
%
%   See also UNIFORM_NYSTROM, GREEDY_CHOLESKY, ACC_RPCHOLESKY.

n        = K.n;
oversamp = log(s);
kk       = ceil(s / (4*oversamp));
nlev     = ceil(log(n/s) / log(2));
perm     = randperm(n);

sz    = zeros(1, nlev+1);
sz(1) = n;
for l = 2:nlev+1
    sz(l) = ceil(sz(l-1)/2);
end

samp  = (1:sz(end))';
ind   = perm(samp);
w     = ones(numel(ind), 1);
kdiag = K.diagonal();

for l = nlev:-1:1
    cur = perm(1:sz(l));
    KS  = K.block(cur, ind);                    % current subset x landmarks
    SKS = KS(samp, :);
    m   = size(SKS, 1);

    if kk >= m
        lam = 10e-6;
    else
        ev  = sort(eig((SKS .* w) .* w'), 'descend');   % diag(w)*SKS*diag(w)
        lam = (sum(diag(SKS) .* w.^2) - sum(abs(ev(1:kk)))) / kk;
    end
    if ~(lam > 0)
        lam = 10e-6;
    end

    R    = KS / (SKS + diag(lam * w.^(-2)));
    rres = max(0, kdiag(cur) - sum(R .* KS, 2));

    if l ~= 1
        lev  = min(1, oversamp / lam * rres);
        samp = find(rand(sz(l), 1) < lev);
        if isempty(samp)
            lev(:) = s / sz(l);
            samp   = randsample(sz(l), s);
        end
        w = sqrt(1 ./ lev(samp));
    else
        lev = min(1, rres / lam);
        if nnz(lev) < s                         % roundoff guard; see header
            lev = lev + eps * max(lev);
        end
        samp = datasample(1:n, s, 'Replace', false, 'Weights', lev)';
    end
    ind = perm(samp);
end

S = ind(:);
C = K.cols(S);

end
