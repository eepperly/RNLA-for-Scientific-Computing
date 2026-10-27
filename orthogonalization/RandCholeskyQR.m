function [Q, R] = RandCholeskyQR(A, k, zeta)
%RANDCHOLESKYQR  Thin QR by whitening with a sketch, then CholeskyQR.
%   [Q,R] = RANDCHOLESKYQR(A) returns a thin QR factorization of the tall-thin
%   m-by-n matrix A, with Q orthonormal to working precision and A = Q*R.
%
%   [Q,R] = RANDCHOLESKYQR(A,k) uses a sketch of k rows (default 2n).
%   [Q,R] = RANDCHOLESKYQR(A,k,zeta) sets the SparseStack sparsity (default 4).
%
%   This is the randomized CholeskyQR algorithm of the survey:
%
%       1. Draw a subspace embedding Omega, m-by-k
%       2. Sketch  Ahat := Omega'*A
%       3. Householder QR of the sketch, Ahat = Q_Omega * R_Omega
%       4. Whiten, B := A / R_Omega
%       5. Gram matrix G := B'*B
%       6. Cholesky G = R_G'*R_G
%       7. Q := B / R_G and R := R_G * R_Omega
%
%   Steps 5-7 are exactly CholeskyQR, which is unstable on an ill-conditioned
%   input.  It is safe here because step 4 has already made B well conditioned:
%   Omega is a subspace embedding for range(A), so kappa(B) = O(1) whatever
%   kappa(A) is, and the Gram matrix squaring that sinks plain CholeskyQR
%   squares something close to 1.  The randomness buys the conditioning, not
%   the accuracy.
%
%   The sketch is a SparseStack embedding, so step 2 costs O(zeta*nnz(A))
%   rather than a dense product.
%
%   See also CHOLESKYQR, CHOLESKYQR2, SPARSESTACK.

[m, n] = size(A);
if m < n
    error('RandCholeskyQR:NotTall', 'A must be tall: got %d-by-%d.', m, n);
end
if nargin < 2 || isempty(k),    k    = 2*n; end
if nargin < 3 || isempty(zeta), zeta = 4;   end
validateattributes(k,    {'numeric'}, {'scalar','integer','positive'}, mfilename, 'k',    2);
validateattributes(zeta, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'zeta', 3);
if k < n
    error('RandCholeskyQR:SketchTooSmall', ...
          'Sketch size k = %d must be at least n = %d.', k, n);
end
k = min(k, m);

Ahat     = SparseStack(A, [], k, zeta);     % 2. Omega'*A, k-by-n
[~, Rom] = qr(Ahat, 0);                     % 3. Householder QR of the sketch
B        = A / Rom;                         % 4. whitened basis, kappa(B) = O(1)

G          = B' * B;                        % 5.
[Rg, flag] = chol(G);                       % 6.
if flag ~= 0
    error('RandCholeskyQR:NotPositiveDefinite', ...
          ['Whitened Gram matrix is not numerically positive definite ' ...
           '(leading minor %d); the sketch failed to embed range(A). ' ...
           'Try a larger k.'], flag);
end

Q = B / Rg;                                 % 7.
R = Rg * Rom;

end
