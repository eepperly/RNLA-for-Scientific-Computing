function [S, d] = SelectLeveragedVolume(U, s)
%SELECTLEVERAGEDVOLUME  Index selection by leveraged volume sampling.
%   S = SELECTLEVERAGEDVOLUME(U,s) returns s row indices of the n-by-k matrix U
%   with orthonormal columns, following the leveraged volume sampling of
%
%     M. Derezinski and M. K. Warmuth, "Leveraged volume sampling for linear
%     regression", NeurIPS 2018,
%
%   whose distribution over subsets of size s is
%
%       P{S} proportional to det(U(S,:)' * Q_S^{-1} * U(S,:)) * prod_{i in S} q_i,
%
%   with q_i = ell_i/k the leverage score distribution and Q_S = diag(s*q_i).
%
%   AT s = k THIS IS ORDINARY VOLUME SAMPLING.  With U(S,:) square,
%   det(U(S,:)'Q_S^{-1}U(S,:)) = det(U(S,:))^2 / prod_{i in S}(s*q_i), and the
%   prod q_i factor cancels it down to det(U(S,:))^2 / s^k -- a constant
%   multiple of the volume sampling weight.  So the leverage rescaling only
%   changes anything once s > k, and at s = k this function and SELECTVOLUME
%   sample from the same distribution.  Seeing the two agree there is a useful
%   check on both.
%
%   OVERSAMPLING.  For s > k the k-row core is volume sampled and the remaining
%   s-k indices are drawn from the leverage distribution over the rows not yet
%   chosen -- the leverage weighting being what distinguishes this from the
%   uniform top-up in SELECTVOLUME.  This is a practical construction chosen to
%   agree with the exact distribution at s = k; sampling the true s > k
%   distribution needs the determinantal rejection sampler of the reference
%   above, which is not implemented here.
%
%   See also SELECTVOLUME, REJECTIONRPQR, SELECTLEVERAGE.

[n, k] = size(U);
if s > n
    error('SelectLeveragedVolume:TooMany', ...
          's = %d exceeds the %d rows of U.', s, n);
end

S = RejectionRPQR(U);                       % exactly k indices, VS_k(U)

if s > k
    ell   = sum(U.^2, 2);
    rest  = setdiff((1:n)', S);
    extra = rest(sample_no_replace(ell(rest), s-k));    % leverage top-up
    S     = [S; extra];
end

S = sort(S(1:min(s,numel(S))));

% Importance weights of the leverage distribution, the D of the weighted DEIM
% constant.  Applied to all s rows, including the volume-sampled core.
ell = sum(U.^2, 2);
d   = sqrt(k ./ (s * ell(S)));

end
