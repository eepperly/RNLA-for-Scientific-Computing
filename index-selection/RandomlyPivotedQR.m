function S = RandomlyPivotedQR(M, k)
%RANDOMLYPIVOTEDQR  Randomly pivoted QR, direct implementation.
%   S = RANDOMLYPIVOTEDQR(M,k) selects k column indices of M by the adaptive
%   sampling procedure (2.2) of
%
%     E. N. Epperly, "Adaptive randomized pivoting and volume sampling" (2025):
%
%     1. draw a column j with probability ||M(:,j)||^2 / ||M||_F^2,
%     2. orthogonalize all of M against that column,
%
%   repeated k times.  This is the plain version: every iteration touches the
%   whole matrix, costing O(k * numel(M)).  ACCELERATEDRPQR samples from the
%   same distribution using rejection sampling and block arithmetic instead,
%   and is the one to use when M is large.
%
%   See also ACCELERATEDRPQR, CPQR_PARTIAL.

n = size(M,2);
S = zeros(k,1);
c = sum(M.^2, 1);                   % squared column norms

for i = 1:k
    if all(c <= 0)
        S = S(1:i-1);
        return
    end
    j    = sample_no_replace_weighted(c);
    S(i) = j;

    v = M(:,j) / norm(M(:,j));
    M = M - v*(v'*M);               % orthogonalize everything against column j
    c = max(sum(M.^2, 1), 0);
    c(S(1:i)) = 0;
end

end

% -------------------------------------------------------------------------

function j = sample_no_replace_weighted(c)
%SAMPLE_NO_REPLACE_WEIGHTED  One draw with probability proportional to c.
cw = cumsum(c(:));
j  = find(cw >= rand()*cw(end), 1, 'first');
end
