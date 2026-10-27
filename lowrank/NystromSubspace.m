function [U, Lambda] = NystromSubspace(A, k, q, n)
%NYSTROMSUBSPACE  Randomized Nystrom approximation with subspace iteration.
%   [U,Lambda] = NYSTROMSUBSPACE(A,k) returns a rank-k approximation
%   A ~ U*diag(Lambda)*U' of the n-by-n positive semidefinite matrix A, using
%   q = 2 steps of subspace (power) iteration to improve the test matrix.
%   Lambda is nonnegative and in descending order.
%
%   [U,Lambda] = NYSTROMSUBSPACE(A,k,q) takes q iterations.
%
%   Following the Tropp-Webber survey, the test matrix is
%
%       Omega = orth(A^q * G),      G standard Gaussian,
%
%   built by alternating a product with A and a QR orthonormalization.  Each
%   step raises the spectrum to a higher power, so the gap between the wanted
%   and unwanted eigenvalues widens geometrically and a random subspace is
%   pushed toward the leading invariant subspace.  Orthonormalizing between
%   steps is not optional: without it the columns collapse onto the dominant
%   eigenvector in floating point.
%
%   Cost is q+1 products with A -- q for the iteration, one inside the Nystrom
%   formula -- against the single pass of NYSTROM.  For a fixed budget of
%   matrix products, NYSTROMKRYLOV converges faster, because it keeps every
%   block it computes instead of discarding all but the last.
%
%   A may be an explicit matrix, or a function handle computing X -> A*X, in
%   which case the dimension n must be passed as a fourth argument.
%
%   See also NYSTROM, NYSTROMKRYLOV, NYSTROM_FACTOR.

if nargin < 3 || isempty(q)
    q = 2;
end
validateattributes(k, {'numeric'}, {'scalar','integer','positive'},    mfilename, 'k', 2);
validateattributes(q, {'numeric'}, {'scalar','integer','nonnegative'}, mfilename, 'q', 3);

if nargin < 4, n = []; end
[Afun, n] = as_operator(A, n);
if k > n
    error('NystromSubspace:RankTooLarge', ...
          'Rank k = %d exceeds the dimension n = %d.', k, n);
end

[Omega, ~] = qr(randn(n,k), 0);
for i = 1:q
    [Omega, ~] = qr(Afun(Omega), 0);
end

[U, Lambda] = nystrom_factor(Afun, Omega);

end
