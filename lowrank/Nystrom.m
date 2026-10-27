function [U, Lambda] = Nystrom(A, k, n)
%NYSTROM  Single-pass randomized Nystrom approximation of a PSD matrix.
%   [U,Lambda] = NYSTROM(A,k) returns a rank-k approximation
%   A ~ U*diag(Lambda)*U' of the n-by-n positive semidefinite matrix A, from a
%   single pass over A.  Lambda is nonnegative and in descending order.
%
%   The test matrix is standard Gaussian, and the approximation is
%
%       A  ~  (A*Omega) * (Omega'*A*Omega)^+ * (A*Omega)'.
%
%   One product A*Omega is all the access to A that is required, so this works
%   in the streaming setting where A is seen once and cannot be revisited.
%
%   Accuracy is limited by how well a random k-dimensional subspace captures
%   the top eigenvectors of A.  For a flat spectrum that is poor; when A has a
%   slowly decaying spectrum, spend extra passes with NYSTROMSUBSPACE or
%   NYSTROMKRYLOV rather than enlarging k.
%
%   A may be an explicit matrix, or a function handle computing X -> A*X, in
%   which case the dimension n must be passed as a third argument.
%
%   Example:
%       [U,Lam] = Nystrom(A, 100);
%       err = norm(A - U*diag(Lam)*U');
%
%   See also NYSTROMSUBSPACE, NYSTROMKRYLOV, NYSTROM_FACTOR.

validateattributes(k, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'k', 2);
if nargin < 3, n = []; end
[Afun, n] = as_operator(A, n);
if k > n
    error('Nystrom:RankTooLarge', ...
          'Rank k = %d exceeds the dimension n = %d.', k, n);
end

Omega = randn(n, k);
[U, Lambda] = nystrom_factor(Afun, Omega);

end
