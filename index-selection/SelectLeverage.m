function [S, d] = SelectLeverage(U, s)
%SELECTLEVERAGE  Index selection by leverage score sampling.
%   S = SELECTLEVERAGE(U,s) draws s distinct row indices of the n-by-k matrix U
%   with probability proportional to the leverage scores
%
%       ell_j = ||U(j,:)||^2,      sum_j ell_j = k,
%
%   which for a matrix with orthonormal columns are just its squared row norms,
%   available in one O(nk) pass.
%
%   Sampling is without replacement, as DEIM needs distinct indices; see
%   SAMPLE_NO_REPLACE.  This biases the draw slightly relative to i.i.d.
%   leverage sampling, but i.i.d. sampling would produce repeats and a
%   singular U(S,:).
%
%   Leverage scores are marginals: they say how much each row matters on its
%   own, but not how rows interact, so nothing stops the sample from piling up
%   on several rows carrying the same direction.  Volume sampling fixes exactly
%   that, at the cost of being adaptive.
%
%   The weights d_i = sqrt(k/(s*ell_i)) are the importance weights for this
%   distribution: rows drawn because their leverage is high are damped back
%   down, which is what keeps the weighted least-squares fit unbiased.
%
%   See also SELECTVOLUME, SELECTLEVERAGEDVOLUME, SELECTCPQR, SELECTUNIFORM.

k   = size(U,2);
ell = sum(U.^2, 2);
S   = sort(sample_no_replace(ell, s));
d   = sqrt(k ./ (s * ell(S)));

end
