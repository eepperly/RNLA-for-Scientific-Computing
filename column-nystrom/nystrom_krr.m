function beta = nystrom_krr(C, S, y, lam)
%NYSTROM_KRR  Kernel ridge regression restricted to column Nystrom landmarks.
%   beta = NYSTROM_KRR(C,S,y,lam) fits f(x) = sum_{j in S} beta_j k(x, x_j) by
%
%       minimize  ||C*beta - y||^2 + lam*n * beta' * A(S,S) * beta,
%
%   with C = A(:,S) the landmark columns, n = size(C,1), and A(S,S) = C(S,:).
%   This is the KRR objective (1/n)||f(X) - y||^2 + lam ||f||_H^2 over the span
%   of the landmarks -- the standard way a column Nystrom approximation enters
%   KRR (Rudi, Camoriano and Rosasco, "Less is more", NeurIPS 2015).  It uses
%   only the columns the selection method already evaluated, and the same lam
%   normalization as exact KRR, (A + lam*n*I) alpha = y.
%
%   The normal equations (C'C + lam n A(S,S)) beta = C'y are not formed.  The
%   problem is solved as the least-squares problem
%
%       [C; sqrt(lam n) W^{1/2}] beta ~ [y; 0],    W = A(S,S),
%
%   by a minimum-norm solver, with W^{1/2} from a clipped eigendecomposition.
%   If A(S,S) is singular (e.g. near-duplicate landmarks) the minimizer is not
%   unique, but every minimizer gives the same predictions: for a psd kernel,
%   A(S,S) v = 0 forces k(x, x_S) v = 0 for every x.
%
%   The same solver is used for every selection method.
%
%   See also KERNELORACLE.

n = size(C, 1);
s = numel(S);
W = C(S, :);
W = (W + W') / 2;
[V, E] = eig(W);
Rw = sqrt(max(diag(E), 0)) .* V';               % W = Rw'*Rw

beta = lsqminnorm([C; sqrt(lam*n) * Rw], [y; zeros(s, 1)]);

end
