function [est, M] = xnystrace(Afun, n, M)
%XNYSTRACE  XNysTrace exchangeable Nystrom trace estimator, with resphering.
%   est = XNYSTRACE(Afun,n,M) estimates tr(A) for a positive semidefinite A
%   from a single block of M matvecs, averaging the leave-one-out estimates
%
%       est_i = tr(Ahat_(i)) + gamma_i' * (A - Ahat_(i)) * gamma_i,
%       Ahat_(i) = A<Omega_{-i}>,
%
%   where gamma_i is omega_i resphered onto range(Omega_{-i})^perp, the
%   subspace the Nystrom residual lives on (it annihilates Omega_{-i}):
%
%       gamma_i = sqrt(n-M+1) * (I - P_i) omega_i / ||(I - P_i) omega_i||,
%       P_i     = orthogonal projector onto range(Omega_{-i}).
%
%   IMPLEMENTATION.  With Y = A*Omega and Gm = Omega'*Y, removing one column is
%   a Schur-complement rank-one downdate:
%
%       Ahat - Ahat_(i) = z_i z_i' / (Gm^{-1})_ii,      z_i = Y Gm^{-1} e_i,
%
%   and because Nystrom interpolates A on range(Omega) the residual quadratic
%   form collapses to  omega_i'(A - Ahat_(i))omega_i = 1/(Gm^{-1})_ii.  The
%   resphering norm is the classical leave-one-out residual
%   ||(I - P_i) omega_i||^2 = 1/((Omega'Omega)^{-1})_ii.  So every estimate
%   comes from M-by-M quantities, and the method needs one pass over A.
%
%   As in NYSTROMPP the computation is carried out exactly on A + nu*I, whose
%   core is safely positive definite, and nu*n is subtracted at the end, so the
%   stabilizing shift costs no bias.
%
%   [est,nmv] = XNYSTRACE(...) also returns the matvecs used, M.
%
%   See also XTRACE, NYSTROMPP, RESPHERE.

m = M;

Om = resphere([], n, m);
Y  = Afun(Om);                              % m matvecs, single pass

nu = sqrt(n) * eps * norm(Y);
Y  = Y + nu*Om;                             % work with A + nu*I
C  = shifted_chol(Om' * Y);                 % Gm = C'*C

B  = Y / C;                                 % tr(Ahat) = ||B||_F^2
Ci = C \ eye(m);                            % C^{-1}
d  = sum(Ci.^2, 2)';                        % (Gm^{-1})_ii
Zm = B * Ci';                               % z_i = Y Gm^{-1} e_i

lowrank = norm(B, 'fro')^2 - vecnorm(Zm).^2 ./ d;  % tr(Ahat_(i))
quad    = 1 ./ d;                                   % omega_i'(A - Ahat_(i))omega_i

[~, RO] = qr(Om, 0);
ROi     = RO \ eye(m);
scale   = (n - m + 1) * sum(ROi.^2, 2)';            % resphering, (n-m+1)/||(I-P_i)omega_i||^2

est = mean(lowrank + scale .* quad) - nu*n;

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
