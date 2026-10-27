function [Q, R] = CholeskyQR2(A)
%CHOLESKYQR2  CholeskyQR applied twice, with a shifted first pass.
%   [Q,R] = CHOLESKYQR2(A) runs a shifted CholeskyQR on A, then a plain
%   CholeskyQR on the result:
%
%       G  = A'*A;  R1 = chol(G + trace(G)*eps*I);  Q1 = A/R1;
%       [Q,R2] = CholeskyQR(Q1);  R = R2*R1.
%
%   Without the shift the first Cholesky fails outright once kappa(A) exceeds
%   about eps^{-1/2} ~ 1e8, because the Gram matrix squares the condition
%   number and stops being numerically positive definite.  The diagonal shift
%   trace(G)*eps pushes it back inside the cone, so the factorization always
%   succeeds and the method returns something for every input.
%
%   Succeeding is not the same as being accurate.  The shift is what the
%   smallest singular values get replaced by, so Q1 has condition number about
%   (trace(G)*eps)^{-1/2} once sigma_min^2 falls below the shift -- roughly
%   1e8 regardless of how ill conditioned A actually was.  The second pass
%   then leaves an orthogonality error of order kappa(Q1)^2*eps, which is
%   O(1).  Recovering full accuracy from a shifted start needs a third pass
%   (shifted CholeskyQR3); two are enough only while kappa(A) <~ 1e8.
%
%   The first pass is always shifted, since that is where the breakdown is.
%   The second pass tries a plain Cholesky first and only falls back to the
%   shift if that fails: shifting it unconditionally would add a perturbation
%   of order n*eps and cap the attainable orthogonality for no benefit, but
%   without the fallback the second pass itself fails once kappa(A) reaches
%   about 1e16, because Q1 by then has kappa ~ 1e8 and its Gram matrix is
%   again borderline.
%
%   See also CHOLESKYQR, RANDCHOLESKYQR.

n = size(A,2);

G = A' * A;
s = trace(G) * eps;                     % diagonal shift
[R1, flag] = chol(G + s*eye(n));
if flag ~= 0
    error('CholeskyQR2:NotPositiveDefinite', ...
          ['Shifted Gram matrix is still not numerically positive definite ' ...
           '(leading minor %d).'], flag);
end
Q1 = A / R1;

% Second pass: plain where possible, shifted where necessary.
G2 = Q1' * Q1;
[R2, flag] = chol(G2);
if flag ~= 0
    [R2, flag] = chol(G2 + trace(G2)*eps*eye(n));
    if flag ~= 0
        error('CholeskyQR2:NotPositiveDefinite', ...
              ['Shifted Gram matrix is still not numerically positive ' ...
               'definite on the second pass (leading minor %d).'], flag);
    end
end
Q = Q1 / R2;
R = R2 * R1;

end
