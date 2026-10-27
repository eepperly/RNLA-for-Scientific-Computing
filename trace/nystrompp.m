function [est, M] = nystrompp(Afun, n, M, t)
%NYSTROMPP  Nystrom++ trace estimator with resphered probes, for PSD A.
%   est = NYSTROMPP(Afun,n,M) estimates tr(A) for a positive semidefinite A
%   using M matvecs, split as t = round(M/2) for the Nystrom sketch and
%   s = M - t probes for the residual:
%
%       Ahat = A<Omega> = (A*Omega) (Omega'*A*Omega)^+ (A*Omega)',
%       est  = tr(Ahat) + (1/s) sum_i gamma_i' * (A - Ahat) * gamma_i.
%
%   RESPHERING.  The Nystrom residual A - Ahat annihilates Omega, and being
%   symmetric it therefore lives on range(Omega)^perp.  So the probes are
%   resphered onto that complement: uniform on the sphere of radius sqrt(n-t)
%   in range(Omega)^perp.  Note the subspace is range(Omega), not range(A*Omega)
%   as in Hutch++.  Because Omega is known before any matvec, the probes can be
%   generated up front and the method keeps its single-pass structure: A is
%   applied once, to the block [Omega, Gamma].
%
%   STABILITY.  The core Omega'*A*Omega is badly conditioned under spectral
%   decay, so the estimator is run on the shifted matrix A + nu*I, whose core
%   is safely positive definite and factored by Cholesky, and nu*n is
%   subtracted at the end.  That is exact Nystrom++ for A + nu*I, so the shift
%   costs no bias.
%
%   est = NYSTROMPP(Afun,n,M,t) sets the sketch size.  [est,M] = ... also
%   returns the matvecs used, t + s.
%
%   See also RESPHERE, XNYSTRACE, HUTCHPP.

if nargin < 4 || isempty(t)
    t = max(1, round(M/2));
end
s = M - t;
if s < 1
    error('nystrompp:Budget', 'Budget M = %d leaves no probes for t = %d.', M, t);
end

Om      = randn(n, t);
[QO, ~] = qr(Om, 0);
G       = resphere(QO, n, s);               % resphered, in range(Omega)^perp

AOG = Afun([Om G]);                         % t + s matvecs, single pass
Y   = AOG(:, 1:t);
Z   = AOG(:, t+1:end);

% Shift to A + nu*I.
nu = sqrt(n) * eps * norm(Y);
Y  = Y + nu*Om;
Z  = Z + nu*G;
C  = shifted_chol(Om'*Y);

B      = Y / C;                             % tr(Ahat) = ||Y C^{-1}||_F^2
trAhat = norm(B, 'fro')^2;

% gamma'*(A_nu - Ahat_nu)*gamma = gamma'*z - ||C^{-T} Y' gamma||^2
F    = (C') \ (Y' * G);
quad = sum(G .* Z, 1) - sum(F.^2, 1);

est = trAhat + mean(quad) - nu*n;

end

function C = shifted_chol(Gm)
%SHIFTED_CHOL  Cholesky of a symmetric matrix that should be positive definite,
%   nudging the diagonal if rounding has left it marginally indefinite.
Gm = (Gm + Gm')/2;
[C, flag] = chol(Gm);
bump = eps * norm(Gm, 'fro');
while flag ~= 0
    Gm = Gm + bump * eye(size(Gm));
    bump = 10 * bump;
    [C, flag] = chol(Gm);
end
end
