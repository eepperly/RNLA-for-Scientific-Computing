function ell = estimate_leverage(A, zeta, kproj)
%ESTIMATE_LEVERAGE  Fast estimate of the leverage scores of a tall matrix.
%   ell = ESTIMATE_LEVERAGE(A) returns an m-by-1 vector of estimated leverage
%   scores for the m-by-n matrix A (m >= n).  The exact scores are the squared
%   row norms of any orthonormal basis for range(A); this routine estimates
%   them without forming one, in two stages:
%
%     1. Sketched whitening.  Draw an m-by-2n sparse stack Om with zeta = 4
%        and take the QR factorization Om'*A = Q*R.  Because Om is a subspace
%        embedding, A*inv(R) has nearly orthonormal columns, so the squared
%        row norms of A*inv(R) approximate the leverage scores.
%
%     2. Johnson-Lindenstrauss projection.  Rather than form the m-by-n matrix
%        A*inv(R), post-multiply by an n-by-kproj matrix Psi of random +/-1
%        signs scaled by 1/sqrt(kproj), with kproj = ceil(log(m)).  Since
%        E[Psi*Psi'] = I, the squared row norms of A*(R\Psi) are unbiased
%        estimates of those of A*inv(R), at a cost of kproj columns, not n.
%
%   ell = ESTIMATE_LEVERAGE(A,zeta,kproj) overrides the sparse stack sparsity
%   zeta (default 4) and the projection width kproj (default ceil(log(m))).
%
%   The estimates are noisy and biased upward, both by design.  The noise is
%   from the JL projection: relative standard deviation roughly sqrt(2/kproj),
%   so with the default the scores are good only to a constant factor.  The
%   bias is from the 2n-column sketch, which whitens imperfectly: sum(ell)
%   estimates sum_i 1/sigma_i(Om'*Q)^2 rather than n, measured at about 2x for
%   the default width (and 1.34x at 4n, 1.15x at 8n, 1.03x at 32n).  Raising
%   kproj cuts the variance but not this bias.
%
%   Both are harmless here.  Leverage score sampling needs only a
%   constant-factor overestimate of each score; overshooting costs a few extra
%   samples, never correctness.
%
%   Cost is one sparse sketch, one small QR, and one m-by-n times n-by-kproj
%   product -- O(nnz(A) + n^3 + m*n*kproj), versus O(m*n^2) for the exact
%   scores.
%
%   See also LEVERAGESAMPLING, SPARSESTACK.

[m, n] = size(A);
if m < n
    error('estimate_leverage:NotTall', 'A must be tall: got %d-by-%d.', m, n);
end
if nargin < 2 || isempty(zeta),  zeta  = 4;              end
if nargin < 3 || isempty(kproj), kproj = ceil(log(m));   end
validateattributes(zeta,  {'numeric'}, {'scalar','integer','positive'}, mfilename, 'zeta',  2);
validateattributes(kproj, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'kproj', 3);

% Stage 1: sketch and whiten.  triu() takes the upper triangle so this works
% whether qr returns R or the Householder-packed form.
% full() so a sparse A still yields a dense n-by-n R; the sketch is small.
R = triu(qr(full(SparseStack(A, [], min(2*n, m), zeta)), 0));

if rcond(R) < eps
    warning('estimate_leverage:RankDeficient', ...
            'Sketched R is numerically singular (rcond = %.2e); estimates may be unreliable.', rcond(R));
end

% Stage 2: project onto kproj dimensions and take squared row norms.
Psi = (2*randi(2, n, kproj) - 3) / sqrt(kproj);
ell = sum((A * (R \ Psi)).^2, 2);

end
