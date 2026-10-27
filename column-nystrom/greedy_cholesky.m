function [S, C, F] = greedy_cholesky(K, k)
%GREEDY_CHOLESKY  Column Nystrom landmarks by greedy pivoted Cholesky.
%   [S,C,F] = GREEDY_CHOLESKY(K,k) runs k steps of partial Cholesky with
%   complete (diagonal) pivoting: each step pivots on the largest entry of the
%   residual diagonal.
%   Returns the pivots S, the columns C = A(:,S), and the factor F with
%   A<S> = F*F'.  K is a KERNELORACLE.
%
%   This is the standard one-column-at-a-time algorithm, following greedy_chol
%   in the thesis code.  Greedy pivoting cannot be blocked without changing
%   which pivots it picks, so there is no faster variant with the same output.
%
%   Entry accesses: n for the diagonal plus n per pivot, (k+1)*n in total.
%
%   See also ACC_RPCHOLESKY, UNIFORM_NYSTROM, RLS_NYSTROM.

n = K.n;
d = K.diagonal();
F = zeros(n, k);
C = zeros(n, k);
S = zeros(k, 1);

for i = 1:k
    [dmax, s] = max(d);
    if dmax <= 0                                % residual exhausted
        S = S(1:i-1);  C = C(:,1:i-1);  F = F(:,1:i-1);
        return
    end
    c = K.cols(s);
    g = c - F(:,1:i-1) * F(s,1:i-1)';          % residual column
    F(:,i) = g / sqrt(g(s));
    C(:,i) = c;
    S(i)   = s;
    d = max(d - F(:,i).^2, 0);                  % clip roundoff
end

end
