function c = cheb_moments(A, n, m, K)
%CHEB_MOMENTS  Stochastic estimation of Chebyshev moments.
%   c = CHEB_MOMENTS(A,n,m,K) returns estimates c(k) of the Chebyshev moments
%   c_k(A) = tr(T_k(A)) for k = 1,...,K, where T_k is the Chebyshev polynomial
%   of the first kind.  A must be symmetric with spectrum in [-1,1], and K must
%   be even.
%
%   m probes are drawn uniformly from the sphere of radius sqrt(n) and advanced
%   by the Chebyshev three-term recurrence, so V_k = T_k(A)*Omega.  The moments
%   are then Girard-Hutchinson estimates of tr(T_k(A)).
%
%   Only K/2 products with A are taken, not K, via the doubling identities
%
%       T_{2k}   = 2 T_k^2 - T_0,       T_{2k-1} = 2 T_k T_{k-1} - T_1,
%
%   which recover the odd and even moments up to K from inner products among
%   the blocks already computed.
%
%   A may be an explicit matrix or a function handle computing X -> A*X.
%
%   See also KPM, CHEB_MOMENT_REGRESSION.

if isa(A,'function_handle'), Afun = A; else, Afun = @(X) A*X; end
if mod(K,2) ~= 0
    error('cheb_moments:OddDegree', 'K must be even; got %d.', K);
end

Omega = randn(n, m);
Omega = sqrt(n) * Omega ./ vecnorm(Omega);      % sphere of radius sqrt(n)

c = zeros(K,1);

V0 = Omega;
V1 = Afun(Omega);

c01  = sum(V0(:) .* V1(:));                     % <V_0, V_1>_F, reused below
c(1) = c01 / m;
c(2) = 2/m * sum(V1(:).^2) - n;

Vkm1 = V0;
Vk   = V1;
for k = 2:K/2
    Vnew = 2*Afun(Vk) - Vkm1;                   % V_k = T_k(A) Omega
    Vkm1 = Vk;
    Vk   = Vnew;

    c(2*k)   = 2/m * sum(Vk(:).^2) - n;
    c(2*k-1) = (2*sum(Vkm1(:).*Vk(:)) - c01) / m;
end

end
