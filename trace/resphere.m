function G = resphere(Q, n, s)
%RESPHERE  Random probe vectors resphered onto the complement of range(Q).
%   G = RESPHERE(Q,n,s) returns an n-by-s matrix whose columns are
%
%       gamma = sqrt(n - t) * (I - Q*Q') nu / ||(I - Q*Q') nu||,   nu ~ N(0,I),
%
%   with t = size(Q,2), i.e. independent vectors drawn uniformly from the
%   sphere of radius sqrt(n - t) in range(Q)^perp.  Q must have orthonormal
%   columns.  These satisfy E[gamma*gamma'] = I - Q*Q': they are isotropic on
%   the complement, which is all a variance-reduced estimator needs to stay
%   unbiased, and their norm is fixed rather than random, which is where the
%   variance reduction comes from.
%
%   With Q = [] (t = 0) the columns are uniform on the sphere of radius sqrt(n)
%   in R^n, so plain Girard-Hutchinson with spherical probes is the t = 0 case.
%
%   See also HUTCHPP, NYSTROMPP, XTRACE, XNYSTRACE.

t = size(Q,2);
G = randn(n, s);
if t > 0
    G = G - Q*(Q'*G);
    G = G - Q*(Q'*G);           % second pass: projection stays accurate to eps
end
G = sqrt(n - t) * G ./ vecnorm(G);

end
