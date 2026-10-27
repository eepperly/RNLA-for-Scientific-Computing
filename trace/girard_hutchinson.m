function [est, nmv] = girard_hutchinson(Afun, n, M)
%GIRARD_HUTCHINSON  Girard-Hutchinson trace estimator with spherical probes.
%   est = GIRARD_HUTCHINSON(Afun,n,M) estimates tr(A) for the n-by-n symmetric
%   matrix applied by Afun (X -> A*X) using M matvecs:
%
%       est = (1/M) sum_i  omega_i' * A * omega_i.
%
%   Resphering with no low-rank component (t = 0) reduces to drawing the
%   probes uniformly from the sphere of radius sqrt(n), the probe distribution
%   whose variance depends only on the spread of the eigenvalues about their
%   mean.
%
%   [est,nmv] = GIRARD_HUTCHINSON(...) also returns the matvecs used.
%
%   See also RESPHERE, HUTCHPP, XTRACE.

Om  = resphere([], n, M);
est = sum(Om .* Afun(Om), 'all') / M;
nmv = M;

end
