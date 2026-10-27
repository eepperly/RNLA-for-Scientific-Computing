function [S, F] = pivoted_cholesky(A, k, rule)
%PIVOTED_CHOLESKY  Partial Cholesky with random or greedy pivoting, dense A.
%   [S,F] = PIVOTED_CHOLESKY(A,k,rule) runs k steps of pivoted partial
%   Cholesky on the psd matrix A and returns the pivots S and the factor F,
%   A<S> = F*F'.  rule is
%
%       'rp'      RPCholesky: pivot drawn with probability proportional to the
%                 residual diagonal
%       'greedy'  greedy: pivot on the largest residual diagonal entry
%
%   This ports cholesky_helper from rpcholesky.py in the reference code
%   (simple_rpcholesky and greedy), including its choices of normalizing each
%   new column by the square root of the maintained residual diagonal and of
%   clipping that diagonal at zero.  F is stored n-by-k rather than k-by-n.
%
%   See also NUCLEAR_CHOLESKY, ACC_RPCHOLESKY, GREEDY_CHOLESKY.

n = size(A, 1);
d = diag(A);
F = zeros(n, k);
S = zeros(k, 1);

for i = 1:k
    switch rule
        case 'rp',     s = randsample(n, 1, true, d);
        case 'greedy', [~, s] = max(d);
        otherwise,     error('pivoted_cholesky:Rule', 'Unknown rule %s.', rule);
    end
    S(i)   = s;
    F(:,i) = (A(:,s) - F(:,1:i-1) * F(s,1:i-1)') / sqrt(d(s));
    d      = max(d - F(:,i).^2, 0);
end

end
