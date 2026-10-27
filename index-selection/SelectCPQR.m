function [S, d] = SelectCPQR(U, s)
%SELECTCPQR  Index selection by column-pivoted QR (Q-DEIM).
%   S = SELECTCPQR(U,s) returns s row indices of the n-by-k matrix U with
%   orthonormal columns, taken as the first s pivots of MATLAB's column-pivoted
%   QR applied to U'.  This is the Q-DEIM selection of
%
%     Z. Drmac and S. Gugercin, "A new selection operator for the discrete
%     empirical interpolation method", SIAM J. Sci. Comput. 38(2), 2016.
%
%   Deterministic, and the standard strong baseline: its DEIM constant grows
%   far more slowly than the original greedy DEIM ordering.  Included here as
%   the reference the randomized methods have to beat.
%
%   Deterministic selection carries no importance weights, so d is all ones.
%   See also SELECTUNIFORM, SELECTLEVERAGE, SELECTVOLUME, SELECTLEVERAGEDVOLUME.

[~, ~, p] = qr(U', 'vector');
S = sort(p(1:s))';
d = ones(s,1);

end
