function [x, w] = cheb_moment_regression(c, n, q)
%CHEB_MOMENT_REGRESSION  Spectral density estimate by moment regression.
%   [x,w] = CHEB_MOMENT_REGRESSION(c,n) returns a discrete measure
%   muhat = sum_j w(j) delta_{x(j)} on a uniform grid over [-1,1] whose
%   Chebyshev moments come close to the estimates c(1:K), by solving
%
%       minimize ||D (c - M w)||^2   s.t.  w >= 0,  sum(w) = n,
%
%   with M(k,j) = T_k(x_j) and D = diag(1, 1/2, ..., 1/K).
%
%   [x,w] = CHEB_MOMENT_REGRESSION(c,n,q) sets the grid size; the default is
%   q = ceil(K^1.5).
%
%   The 1/k weighting is what makes this a Wasserstein-1 method rather than a
%   generic fit: the error bound of Musco et al. penalises moment k by 1/k^2 in
%   a sum of squares, so the low-order moments, which carry the coarse shape of
%   the spectrum, are matched first.  The constraints w >= 0 and sum(w) = n
%   force the output to be a genuine nonnegative measure of the right mass,
%   which is what lets the same bound control W_1.
%
%   Requires lsqlin from the Optimization Toolbox.
%
%   See also CHEB_MOMENTS, KPM, SLQ.

K = numel(c);
if nargin < 3 || isempty(q)
    q = ceil(K^1.5);
end

x = linspace(-1, 1, q)';

% Moment matrix M(k,j) = T_k(x_j), by the three-term recurrence.
M    = zeros(K, q);
Tkm1 = ones(1,q);               % T_0
Tk   = x';                      % T_1
for k = 1:K
    M(k,:) = Tk;
    Tnew = 2*x'.*Tk - Tkm1;
    Tkm1 = Tk;
    Tk   = Tnew;
end

D = spdiags(1./(1:K)', 0, K, K);

opts = optimoptions('lsqlin','Display','off');
w = lsqlin(D*M, D*c(:), [], [], ones(1,q), n, zeros(q,1), [], [], opts);

end
