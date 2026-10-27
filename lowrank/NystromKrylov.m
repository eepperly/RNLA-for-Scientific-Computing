function [U, Lambda] = NystromKrylov(A, k, q, r, n)
%NYSTROMKRYLOV  Randomized Nystrom approximation with block Krylov iteration.
%   [U,Lambda] = NYSTROMKRYLOV(A,k) returns an approximation
%   A ~ U*diag(Lambda)*U' of the n-by-n positive semidefinite matrix A, built
%   from a block Krylov space of block size k and depth q = 2, then truncated
%   to rank r = k.  Lambda is nonnegative and in descending order.
%
%   [U,Lambda] = NYSTROMKRYLOV(A,k,q,r) sets the depth q and the output rank r.
%   Pass r = [] to keep the full rank (q+1)*k.
%
%   Following the Tropp-Webber survey, the test matrix spans the block Krylov
%   space
%
%       range[ G, A*G, A^2*G, ..., A^q*G ],     G standard Gaussian,
%
%   of dimension (q+1)*k.  Where subspace iteration keeps only the last block,
%   this keeps all of them, so for the same number of products with A it
%   searches a much larger space and converges faster -- the accuracy is
%   governed by a polynomial in A of degree q rather than by the single
%   monomial A^q.  This is the method of choice when the spectrum decays
%   slowly and matrix products are the expensive resource.
%
%   Each new block is orthogonalized against all previous blocks, twice.  The
%   Krylov blocks become numerically dependent very quickly otherwise, which is
%   what makes naive block Krylov implementations fail.
%
%   Cost is q+1 products with A, the same as NYSTROMSUBSPACE with q steps, but
%   the basis is (q+1)*k wide rather than k, so the QR and the Nystrom solve
%   cost correspondingly more.
%
%   A may be an explicit matrix, or a function handle computing X -> A*X, in
%   which case the dimension n must be passed as a fifth argument.
%
%   See also NYSTROM, NYSTROMSUBSPACE, NYSTROM_FACTOR.

if nargin < 3 || isempty(q), q = 2; end
if nargin < 4,               r = k; end
validateattributes(k, {'numeric'}, {'scalar','integer','positive'},    mfilename, 'k', 2);
validateattributes(q, {'numeric'}, {'scalar','integer','nonnegative'}, mfilename, 'q', 3);

if nargin < 5, n = []; end
[Afun, n] = as_operator(A, n);
if (q+1)*k > n
    error('NystromKrylov:BasisTooLarge', ...
          'Krylov basis (q+1)*k = %d exceeds the dimension n = %d.', (q+1)*k, n);
end

% Block Krylov basis, orthonormalized blockwise as it is built.
[Qi, ~] = qr(randn(n,k), 0);
K = Qi;
scale = [];
for i = 1:q
    Z = Afun(Qi);
    if isempty(scale)
        scale = norm(Z, 'fro');         % reference size for the rank test
    end
    Z = Z - K*(K'*Z);
    Z = Z - K*(K'*Z);                   % reorthogonalize; one pass is not enough

    % Pivoted QR so the numerically dependent directions can be dropped.  Once
    % the Krylov space saturates -- A is rank deficient, or the space is
    % already A-invariant -- Z is pure rounding noise, and keeping its columns
    % would put vectors in K that are not orthogonal to the rest of it.  That
    % silently destroys the K'*K = I that makes the shifted Gram matrix in
    % NYSTROM_FACTOR positive definite.
    [Qz, Rz, ~] = qr(Z, 0);
    keep = abs(diag(Rz)) > max(size(Z)) * eps * scale;
    if ~any(keep)
        break                           % happy breakdown: the space is invariant
    end

    Qi = Qz(:, keep);
    K  = [K Qi];                        %#ok<AGROW> grows at most q times
end

[U, Lambda] = nystrom_factor(Afun, K);

% Truncate to the requested rank; Lambda comes back in descending order.
if ~isempty(r)
    validateattributes(r, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'r', 4);
    r = min(r, numel(Lambda));
    U = U(:,1:r);
    Lambda = Lambda(1:r);
end

end
