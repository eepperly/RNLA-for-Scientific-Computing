function [S, F] = nuclear_cholesky(A, k)
%NUCLEAR_CHOLESKY  Partial Cholesky with nuclear score pivoting, dense A.
%   [S,F] = NUCLEAR_CHOLESKY(A,k) runs k steps of pivoted partial Cholesky,
%   each time pivoting on the index with the largest nuclear score of the
%   residual R:
%
%       n_i(R) = ||R(i,:)||^2 / R(i,i),
%
%   the decrease in tr(R) from eliminating index i.  Returns the pivots S and
%   the factor F, A<S> = F*F'.
%
%   As in nuclear_chol in the thesis code, the residual is held explicitly
%   and its column norms recomputed from it at every step, rather than
%   downdated: downdating cancels catastrophically once the residual falls
%   to roundoff relative to A, which on the most compressible matrices
%   happens well before k steps.  Each step costs O(n^2), so this is not a
%   linear-time method; the scores need every entry of A.
%
%   The rank-one update and the norm recomputation are fused and done a block
%   of columns at a time, so no n-by-n temporary is formed.
%
%   NUMERICAL RANK.  Only indices whose residual diagonal exceeds
%   n*eps*max(diag(A)) -- the default tolerance of LAPACK's pivoted Cholesky,
%   xPSTRF -- are eligible, and the method stops early, returning fewer than k
%   pivots, once none are.  Without this, a matrix that is low-rank to working
%   precision breaks the method: once the residual is at roundoff, the score
%   ||R(i,:)||^2 / R(i,i) is a ratio of roundoff, it can select an index whose
%   residual diagonal is essentially zero, and normalizing by its square root
%   blows the factor up.  (Greedy and RPCholesky are protected by pivoting on,
%   or sampling from, the residual diagonal itself.)
%
%   See also PIVOTED_CHOLESKY.

n   = size(A, 1);
tol = n * eps * max(diag(A));                   % xPSTRF default tolerance
R   = A;                                        % residual
F  = zeros(n, k);
S  = zeros(k, 1);
cn = sum(R.^2, 1)';                             % squared column norms
bs = 512;

for i = 1:k
    dg   = diag(R);
    elig = dg > tol;
    elig(S(1:i-1)) = false;
    if ~any(elig)                               % exhausted to working precision
        S = S(1:i-1);
        F = F(:, 1:i-1);
        return
    end
    sc = zeros(n, 1);
    sc(elig) = cn(elig) ./ dg(elig);
    [~, s] = max(sc);

    f = R(:,s) / sqrt(R(s,s));
    F(:,i) = f;
    S(i)   = s;

    for j0 = 1:bs:n                             % R <- R - f*f', norms with it
        J = j0:min(j0+bs-1, n);
        R(:,J) = R(:,J) - f * f(J)';
        cn(J)  = sum(R(:,J).^2, 1)';
    end
end

end
