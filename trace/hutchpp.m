function [est, M] = hutchpp(Afun, n, M, t)
%HUTCHPP  Hutch++ trace estimator with resphered probes.
%   est = HUTCHPP(Afun,n,M) estimates tr(A) using M matvecs with A, split as
%   t = round(M/3) for the rangefinder sketch, t for A*Q, and s = M - 2t probes
%   for the residual:
%
%       Q   = orth(A*Omega),
%       est = tr(Q'*A*Q) + (1/s) sum_i gamma_i' * A * gamma_i.
%
%   The probes gamma_i are resphered (see RESPHERE): uniform on the sphere of
%   radius sqrt(n - t) in range(Q)^perp, so E[gamma*gamma'] = I - Q*Q'.  Since
%   gamma_i is already orthogonal to Q, gamma_i'*A*gamma_i is the residual
%   quadratic form gamma_i'*(I-QQ')*A*(I-QQ')*gamma_i with no explicit
%   projection, and the estimate is unbiased.  Resphering helps most for
%   "signal plus noise" spectra, where the residual eigenvalues are clustered
%   and fixing the probe norm removes most of what variance is left.
%
%   est = HUTCHPP(Afun,n,M,t) sets the sketch size.  [est,M] = ... also
%   returns the matvecs used, M.
%
%   See also RESPHERE, XTRACE, NYSTROMPP.

if nargin < 4 || isempty(t)
    t = max(1, round(M/3));
end
s = M - 2*t;
if s < 1
    error('hutchpp:Budget', 'Budget M = %d leaves no probes for t = %d.', M, t);
end

Om     = randn(n, t);
[Q, ~] = qr(Afun(Om), 0);                   % t matvecs
G      = resphere(Q, n, s);                 % resphered, in range(Q)^perp

AQG = Afun([Q G]);                          % t + s matvecs, one block
est = sum(Q .* AQG(:,1:t), 'all') + sum(G .* AQG(:,t+1:end), 'all') / s;

end
