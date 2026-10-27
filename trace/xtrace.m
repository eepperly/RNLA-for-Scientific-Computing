function [est, nmv] = xtrace(Afun, n, M)
%XTRACE  XTrace exchangeable trace estimator, with resphering.
%   est = XTRACE(Afun,n,M) estimates tr(A) using M matvecs (m = floor(M/2)
%   test vectors, each used twice).  It averages the leave-one-out estimates
%
%       est_i = tr(Q_(i)'*A*Q_(i)) + gamma_i'*A*gamma_i,
%       Q_(i) = orth(A*Omega_{-i}),
%
%   where gamma_i is omega_i resphered onto range(Q_(i))^perp:
%
%       gamma_i = sqrt(n-m+1) * (I - P_i) omega_i / ||(I - P_i) omega_i||,
%       P_i     = Q_(i)*Q_(i)'.
%
%   Since omega_i is uniform on the sphere and independent of Q_(i), the
%   normalized projection is uniform on the unit sphere of the complement, so
%   this is exactly the resphered probe of RESPHERE, built from the
%   left-out vector rather than a fresh draw.
%
%   IMPLEMENTATION.  Only two block matvecs are taken, Y = A*Omega and A*Q
%   with [Q,R] = qr(Y).  Every leave-one-out projector is a rank-one downdate
%   of Q*Q':  range(Y_{-i}) = Q * s_i^perp with s_i = R^{-T} e_i / ||R^{-T} e_i||,
%   because s_i is orthogonal to every column of R but the i-th.  So
%
%       P_i = Q (I - s_i s_i') Q',      tr(Q_(i)'A Q_(i)) = tr(H) - s_i'H s_i,
%
%   with H = Q'*A*Q, and the residual quadratic form and the resphering norm
%   reduce to m-by-m quantities.  Post-processing costs O(n m^2).
%
%   [est,nmv] = XTRACE(...) also returns the matvecs used, 2m.
%
%   See also XNYSTRACE, HUTCHPP, RESPHERE.

m = floor(M/2);

Om     = resphere([], n, m);
Y      = Afun(Om);                          % m matvecs
[Q, R] = qr(Y, 0);
Z      = Afun(Q);                           % m matvecs

H = Q' * Z;                                 % Q'*A*Q
W = Q' * Om;                                % w_i = Q'*omega_i
S = (R') \ eye(m);
S = S ./ vecnorm(S);                        % s_i, the downdate directions

sw = sum(S .* W, 1);                        % s_i'*w_i
V  = W - S .* sw;                           % P_i*omega_i = Q*v_i

% r_i = (I - P_i) omega_i = omega_i - Q v_i,  A r_i = y_i - Z v_i, so
% r_i'A r_i = omega_i'y_i - (Z'omega_i)'v_i - v_i'(Q'y_i) + v_i'H v_i,
% and Q'y_i is the i-th column of R.
T    = Z' * Om;
quad = sum(Om .* Y, 1) - sum(T .* V, 1) - sum(R .* V, 1) + sum(V .* (H*V), 1);

% Resphering: rescale the residual quadratic form so the probe has squared
% norm n - (m-1), the dimension of range(Q_(i))^perp.
rr    = vecnorm(Om).^2 - vecnorm(W).^2 + sw.^2;    % ||(I - P_i) omega_i||^2
scale = (n - m + 1) ./ rr;

lowrank = trace(H) - sum(S .* (H*S), 1);           % tr(Q_(i)'*A*Q_(i))

est = mean(lowrank + scale .* quad);
nmv = 2*m;

end
