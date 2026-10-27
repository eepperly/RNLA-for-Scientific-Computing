function [S, d] = SelectUniform(U, s)
%SELECTUNIFORM  Index selection by uniform sampling without replacement.
%   S = SELECTUNIFORM(U,s) draws s distinct row indices of U uniformly.
%
%   Sampling is without replacement because DEIM needs U(S,:) to have full
%   column rank; a repeated index would make it singular outright.
%
%   The weak baseline.  Uniform sampling ignores U entirely, so it fails
%   whenever the basis is coherent -- exactly the case where the interesting
%   rows are a small minority.
%
%   The weights d = sqrt(n/s) are constant, so D is a multiple of the identity
%   and cancels out of ||(D*U(S,:))^+ D||_2: weighting cannot help uniform
%   sampling, at any oversampling level.
%
%   See also SELECTCPQR, SELECTLEVERAGE, SELECTVOLUME, SELECTLEVERAGEDVOLUME.

n = size(U,1);
S = sort(randsample(n, s, false));
d = sqrt(n/s) * ones(s,1);

end
