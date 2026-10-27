function [Q, R] = CholeskyQR(A)
%CHOLESKYQR  Thin QR by Cholesky of the Gram matrix.
%   [Q,R] = CHOLESKYQR(A) computes G = A'*A, its Cholesky factor G = R'*R, and
%   sets Q = A/R.  In exact arithmetic Q is orthonormal and A = Q*R.
%
%   The Gram matrix squares the condition number, so in floating point the
%   orthogonality loss is
%
%       ||Q'*Q - I|| = O(kappa(A)^2 * eps),
%
%   and once kappa(A) exceeds about eps^{-1/2} ~ 1e8 the Gram matrix is no
%   longer numerically positive definite and the Cholesky fails outright.  That
%   is why plain CholeskyQR is reserved for matrices known to be well
%   conditioned, and why RANDCHOLESKYQR whitens first.
%
%   See also CHOLESKYQR2, RANDCHOLESKYQR.

G = A' * A;
[R, flag] = chol(G);
if flag ~= 0
    error('CholeskyQR:NotPositiveDefinite', ...
          ['Gram matrix is not numerically positive definite (failed at ' ...
           'leading minor %d); A is too ill conditioned for CholeskyQR.'], flag);
end
Q = A / R;

end
