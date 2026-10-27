function T = lanczos_tridiag(A, w, t, reorth)
%LANCZOS_TRIDIAG  Lanczos tridiagonal matrix, with optional reorthogonalization.
%   T = LANCZOS_TRIDIAG(A,w,t) runs t steps of the Lanczos process on the
%   symmetric matrix A with starting vector w and returns the t-by-t symmetric
%   tridiagonal matrix T.  If the process breaks down early, T is returned at
%   the smaller dimension reached, which is the invariant-subspace case and is
%   still exact.
%
%   T = LANCZOS_TRIDIAG(A,w,t,reorth) selects the orthogonalization.  With
%   reorth true (the default) the new vector is projected against every basis
%   vector built so far, twice, at O(n t^2) cost.  With reorth false only the
%   bare three-term recurrence is used, at O(n t) cost.
%
%   The difference is not cosmetic.  In exact arithmetic the three-term
%   recurrence already produces an orthogonal basis, but in floating point
%   orthogonality is lost as soon as a Ritz value converges, and the Lanczos
%   matrix then acquires spurious repeated eigenvalues -- "ghosts" -- that show
%   up in SLQ as duplicated quadrature nodes with split weights.
%
%   A may be an explicit matrix or a function handle computing X -> A*X.
%
%   See also SLQ.

if nargin < 4 || isempty(reorth), reorth = true; end
if isa(A,'function_handle'), Afun = A; else, Afun = @(x) A*x; end

n = numel(w);
Q = zeros(n, t);
alpha = zeros(t,1);
beta  = zeros(t-1,1);

Q(:,1) = w / norm(w);
for j = 1:t
    z = Afun(Q(:,j));
    alpha(j) = Q(:,j)' * z;

    if reorth
        z = z - Q(:,1:j) * (Q(:,1:j)' * z);
        z = z - Q(:,1:j) * (Q(:,1:j)' * z);  % reorthogonalize twice
    else
        z = z - alpha(j) * Q(:,j);           % bare three-term recurrence
        if j > 1
            z = z - beta(j-1) * Q(:,j-1);
        end
    end

    if j == t
        break
    end
    nz = norm(z);
    if nz <= eps * max(1, max(abs(alpha(1:j))))
        t = j;                                % breakdown: invariant subspace
        alpha = alpha(1:t);
        beta  = beta(1:t-1);
        break
    end
    beta(j)  = nz;
    Q(:,j+1) = z / nz;
end

T = diag(alpha) + diag(beta,1) + diag(beta,-1);

end
