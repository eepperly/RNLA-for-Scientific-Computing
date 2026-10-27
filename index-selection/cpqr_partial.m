function p = cpqr_partial(A, k)
%CPQR_PARTIAL  Golub-Businger column-pivoted QR, stopped after k pivots.
%   p = CPQR_PARTIAL(A,k) returns the first k pivot indices of the
%   column-pivoted QR factorization of A, choosing at each step the column of
%   largest residual norm and orthogonalizing the rest against it.
%
%   Written out rather than calling MATLAB's qr for two reasons: qr computes
%   the whole factorization when only k pivots are wanted, which is the
%   difference between O(k*m*n) and O(m*n^2) here, and it would need a working
%   copy of A -- 800 MB for the 10^4-by-10^4 test matrix.  This version touches
%   A read-only and carries only the k-column orthonormal basis.
%
%   The pivots are those of Golub-Businger; the factorization itself is formed
%   by pivoted Gram-Schmidt with reorthogonalization, which selects the same
%   columns.  Squared residual norms are downdated rather than recomputed.
%
%   See also RANDOMLYPIVOTEDQR, ACCELERATEDRPQR.

[m, n] = size(A);
if k > min(m,n)
    error('cpqr_partial:RankTooLarge', 'k = %d exceeds min(size(A)) = %d.', k, min(m,n));
end

c = sum(A.^2, 1);                   % squared residual norms of the columns
p = zeros(k,1);
Q = zeros(m,k);

for j = 1:k
    [~, idx] = max(c);
    p(j) = idx;

    v = A(:,idx);
    if j > 1
        v = v - Q(:,1:j-1)*(Q(:,1:j-1)'*v);
        v = v - Q(:,1:j-1)*(Q(:,1:j-1)'*v);     % reorthogonalize
    end
    nv = norm(v);
    if nv <= eps*sqrt(c(idx))
        p = p(1:j-1);                            % numerically exhausted
        return
    end
    Q(:,j) = v / nv;

    c = max(c - (Q(:,j)'*A).^2, 0);              % downdate
    c(p(1:j)) = 0;                               % never reselect
end

end
