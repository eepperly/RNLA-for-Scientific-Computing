function [U, Lambda] = nystrom_factor(A, Omega)
%NYSTROM_FACTOR  Randomized Nystrom approximation for a given test matrix.
%   [U,Lambda] = NYSTROM_FACTOR(A,Omega) returns the Nystrom approximation
%
%       A  ~  (A*Omega) * (Omega'*A*Omega)^+ * (A*Omega)'  =  U*diag(Lambda)*U'
%
%   of the n-by-n positive semidefinite A, for the n-by-k test matrix Omega.
%   Lambda is returned in descending order and is nonnegative.
%
%   This is the numerically stable formulation from the Tropp-Webber survey
%   (following Tropp, Yurtsever, Udell and Cevher, 2017): the sketch is shifted
%   by nu = eps*norm(Y,'fro') before the Gram matrix is formed, the resulting
%   positive definite matrix is factored by Cholesky rather than inverted, and
%   the shift is subtracted back off the eigenvalues at the end.  Forming
%   Omega'*A*Omega and inverting it directly loses half the digits.
%
%   All three Nystrom variants in this folder reduce to this routine; they
%   differ only in how Omega is built.  The approximation depends on Omega
%   only through its range.
%
%   A may be an explicit matrix or a function handle computing X -> A*X; the
%   dimension is read off Omega, so no size is needed.  Exactly one product
%   with A is taken here.
%
%   See also NYSTROM, NYSTROMSUBSPACE, NYSTROMKRYLOV.

Afun = as_operator(A);

Y  = Afun(Omega);
nu = eps * norm(Y,"fro");            

% The shifted Gram matrix is positive definite in exact arithmetic.  If
% rounding leaves it indefinite, grow the shift and retry rather than failing.
flag = 1;
for attempt = 1:5
    Yv = Y + nu*Omega;
    M  = Omega' * Yv;
    [C, flag] = chol(M);
    if flag == 0
        break
    end
    nu = 10*nu;
end
if flag ~= 0
    error('nystrom_factor:NotPSD', ...
          'Cholesky failed after shifting; is A positive semidefinite?');
end

B = Yv / C;                         % triangular solve, B = Yv*inv(C)
[U, S, ~] = svd(B, 'econ');
Lambda = max(0, diag(S).^2 - nu);   % undo the shift, clip negatives to zero

end
