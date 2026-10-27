function [S, d] = SelectVolume(U, s)
%SELECTVOLUME  Index selection by volume sampling (randomly pivoted QR).
%   S = SELECTVOLUME(U,s) returns s row indices of the n-by-k matrix U with
%   orthonormal columns.  For s = k the sample is drawn exactly from the volume
%   sampling distribution
%
%       P{S} proportional to det(U(S,:))^2,
%
%   using the fast rejection implementation in REJECTIONRPQR.  Volume sampling
%   is the distribution for which adaptive randomized pivoting attains its
%   guarantee; see Epperly, "Adaptive randomized pivoting and volume sampling"
%   (2025), Theorem 1.1.
%
%   Unlike leverage scores, the determinant is a joint quantity: a row that
%   duplicates a direction already chosen contributes nothing to the volume and
%   so is essentially never picked twice.  That is what makes the selected rows
%   spread across the row space rather than clustering.
%
%   OVERSAMPLING.  Volume sampling cannot select more than k rows of a rank-k
%   matrix -- every larger subset has determinant zero.  For s > k the first k
%   indices are volume sampled and the remaining s-k are drawn uniformly from
%   the rows not yet chosen.  That top-up is a practical convention, not part of
%   the volume sampling guarantee.
%
%   See also REJECTIONRPQR, SELECTLEVERAGEDVOLUME, SELECTLEVERAGE, SELECTCPQR.

[n, k] = size(U);
if s > n
    error('SelectVolume:TooMany', 's = %d exceeds the %d rows of U.', s, n);
end

S = RejectionRPQR(U);                       % exactly k indices, VS_k(U)

if s > k
    rest  = setdiff((1:n)', S);
    extra = randsample(rest, s-k, false);   % uniform top-up
    S     = [S; extra];
end

S = sort(S(1:min(s,numel(S))));
d = ones(numel(S),1);      % volume sampling carries no importance weights

end
