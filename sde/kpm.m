function p = kpm(c, n, y)
%KPM  Kernel polynomial method estimate of the spectral density.
%   p = KPM(c,n,y) evaluates the Jackson-damped spectral density estimate
%
%       phat_K(y) = (n/2) w(y) + sum_{k=1}^K beta_k c_k T_k(y) w(y),
%           w(y) = (2/pi) (1-y^2)^{-1/2},
%
%   at the points y, given Chebyshev moment estimates c(1:K) of a matrix with
%   spectrum in [-1,1].  The result is a density on [-1,1] with total mass n,
%   so integrating any f against it estimates tr(f(A)).
%
%   The beta_k are the Jackson damping coefficients
%
%       beta_k = (1/(K+2)) [ (K+2-k) cos(pi k/(K+2))
%                            + sin(pi k/(K+2)) cot(pi/(K+2)) ].
%
%   Damping is not cosmetic.  Truncating the Chebyshev series at K without it
%   produces Gibbs oscillations and a density that goes negative near sharp
%   spectral features; the Jackson weights make phat_K a bona fide nonnegative
%   density.
%
%   The Chebyshev weight w(y) is singular at y = +-1, so p is Inf there.
%
%   See also CHEB_MOMENTS, CHEB_MOMENT_REGRESSION, SLQ.

K = numel(c);
y = y(:);

beta = jackson(K);

% T_k(y) by the three-term recurrence.
w  = (2/pi) ./ sqrt(1 - y.^2);
s  = n/2 * ones(size(y));
Tkm1 = ones(size(y));           % T_0
Tk   = y;                       % T_1
for k = 1:K
    s = s + beta(k) * c(k) * Tk;
    Tnew = 2*y.*Tk - Tkm1;
    Tkm1 = Tk;
    Tk   = Tnew;
end

p = s .* w;

end

function beta = jackson(K)
%JACKSON  Jackson damping coefficients beta_1..beta_K.
k    = (1:K)';
a    = pi/(K+2);
beta = ((K+2-k).*cos(a*k) + sin(a*k)*cot(a)) / (K+2);
end
