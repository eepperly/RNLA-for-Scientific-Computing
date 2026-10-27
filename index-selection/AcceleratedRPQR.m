function S = AcceleratedRPQR(M, k)
%ACCELERATEDRPQR  Randomly pivoted QR, block-wise with rejection sampling.
%   S = ACCELERATEDRPQR(M,k) selects k column indices of M from the same
%   distribution as RANDOMLYPIVOTEDQR, but a block at a time:
%
%     E. N. Epperly, "Adaptive randomized pivoting and volume sampling" (2025),
%     section 4, after Epperly, Moreno and Webber, "Embrace rejection" (2025).
%
%   This follows Algorithm 2.2 of Epperly, Tropp and Webber, "Embrace
%   rejection: kernel matrix approximation by accelerated randomly pivoted
%   Cholesky", SIAM J. Matrix Anal. Appl. 46(4):2527-2557, 2025, translated
%   from RPCholesky on M'*M to QR on M: the residual diagonal of M'*M is the
%   vector of squared residual column norms of M.
%
%   Each sweep proposes a block of columns i.i.d. from the *current* squared
%   residual norms, then walks the block accepting proposal i with probability
%   H(i,i)/u(i), where u = diag(H) on entry and H is deflated by the pivots
%   accepted earlier in the same block.  The first proposal is therefore always
%   accepted, and the rest are rejected exactly when an earlier pivot in the
%   block already explains them.
%
%   The saving over RANDOMLYPIVOTEDQR is arithmetic, not asymptotic: both cost
%   O(k*numel(M)), but this version spends it on matrix-matrix products
%   (U'*M for each accepted block) instead of k rank-one updates of the whole
%   matrix.
%
%   Proposals must come from the current residual norms, not the original
%   column norms.  With fixed proposals the acceptance rate is the ratio of
%   residual to original energy, which for a matrix with fast singular value
%   decay collapses after the first pivot and the sampler stalls.
%
%   The residual norms here are squared, so an exhaustion test on them has to
%   be at eps^2 scale, not eps.  Testing at eps scale stops the sweep while the
%   relative residual is still around 1e-7 and returns far fewer than k pivots.
%
%   See also RANDOMLYPIVOTEDQR, REJECTIONRPQR, CPQR_PARTIAL.

[m, n] = size(M);

c0 = sum(M.^2, 1)';                 % squared column norms; one pass over M
c  = c0;                            % squared residual norms, maintained below
S = zeros(k,1);
ns = 0;
U = zeros(m,0);                     % orthonormal basis of the chosen columns
G = zeros(0,n);                     % U'*M, carried along

maxsweeps = 100 + 20*k;
sweep     = 0;

while ns < k
    sweep = sweep + 1;
    if sweep > maxsweeps
        error('AcceleratedRPQR:NoProgress', ...
              'Rejection sampling selected only %d of %d columns.', ns, k);
    end

    if ~any(c > 0)
        S = S(1:ns);                % residual exactly exhausted
        return
    end

    b = k - ns;
    T = randsample(n, b, true, c);

    C = M(:,T) - U*G(:,T);          % residuals of the proposals
    H = C'*C;

    Tnew = accept_block(H, T);
    if isempty(Tnew), continue, end

    Z  = M(:,Tnew) - U*G(:,Tnew);
    Z  = Z - U*(U'*Z);              % reorthogonalize
    ok = vecnorm(Z) > max(m,n) * eps * sqrt(max(c));
    Z  = Z(:,ok);  Tnew = Tnew(ok);
    if isempty(Tnew), continue, end

    [Qz, ~] = qr(Z, 0);
    Gz = Qz' * M;                                   % the matrix-matrix product
    U  = [U Qz];                                    %#ok<AGROW> at most k wide
    G  = [G; Gz];                                   %#ok<AGROW>

    % Recompute the residual norms from the original ones and G rather than
    % downdating c in place: a single subtraction does not accumulate the error
    % that repeated downdates do.  Neither can resolve a residual below about
    % eps times the column norm, but nothing beyond the numerical rank needs to
    % be resolved -- the pivots there are arbitrary either way.
    c = max(c0 - sum(G.^2, 1)', 0);
    S(ns+1 : ns+numel(Tnew)) = Tnew;
    ns = ns + numel(Tnew);
    c(S(1:ns)) = 0;
end

S = S(1:ns);

end

% -------------------------------------------------------------------------

function Tnew = accept_block(H, T)
%ACCEPT_BLOCK  RejectionSampleSubmatrix, Algorithm 2.1 of Epperly, Tropp and
%   Webber, "Embrace rejection", SIAM J. Matrix Anal. Appl. 46(4):2527-2557,
%   2025.  The acceptance threshold u is diag(H) as it stands on entry, i.e.
%   the residual norms before any deflation within this block, read off H
%   rather than from the incrementally downdated norms carried outside.  The
%   first proposal is therefore always accepted.
b    = numel(T);
u    = diag(H);
Tnew = zeros(0,1);
for i = 1:b
    if H(i,i) > 0 && u(i)*rand() < H(i,i)
        Tnew(end+1,1) = T(i);                       %#ok<AGROW>
        H(i:b, i:b) = H(i:b,i:b) - H(i:b,i)*H(i,i:b)/H(i,i);
    end
end
end
