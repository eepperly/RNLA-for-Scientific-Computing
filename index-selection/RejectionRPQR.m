function S = RejectionRPQR(Q)
%REJECTIONRPQR  Randomly pivoted QR by rejection sampling, for orthonormal Q.
%   S = REJECTIONRPQR(Q) selects k = size(Q,2) row indices of the m-by-k matrix
%   Q, which must have orthonormal columns.  The output is a sample from the
%   volume sampling distribution VS_k(Q) = DPP_k(Q*Q').
%
%   This is the efficient block implementation, Algorithm 4.2 of
%
%     E. N. Epperly, "Adaptive randomized pivoting and volume sampling" (2025),
%
%   with the accept/reject inner loop of its Algorithm 4.1.  Running randomly
%   pivoted QR directly costs a pass over all of Q per pivot, because every
%   iteration reorthogonalizes the whole matrix.  Rejection sampling instead
%   proposes indices from the leverage score distribution and orthogonalizes
%   only the proposals, which is what makes it fast: an expected O(k log k)
%   rejection steps and O(k^3 log k) arithmetic after an O(mk) pass to form the
%   leverage scores.
%
%   The proposal distribution is P{t = j} = ell_j / k with ell_j = ||Q(j,:)||^2,
%   and a proposal t is accepted with probability ||(I-Pi)Q(t,:)'||^2 / ell_t,
%   where Pi projects onto the rows already chosen.  An index already selected
%   has zero residual and so is never re-accepted, which is what keeps S a set.
%
%   See also SELECTVOLUME, SELECTLEVERAGE, SELECTCPQR.

[m, k] = size(Q);
ell = sum(Q.^2, 2);                 % leverage scores; these sum to k

S   = zeros(k,1);
ns  = 0;                            % number selected so far
U   = zeros(k,0);                   % orthonormal basis for the chosen rows

maxsweeps = 100 + 20*k;             % guard; the expected count is O(log k)
sweep     = 0;

while ns < k
    sweep = sweep + 1;
    if sweep > maxsweeps
        error('RejectionRPQR:NoProgress', ...
              'Rejection sampling selected only %d of %d indices.', ns, k);
    end

    % Propose a block of k indices from the leverage score distribution.
    T = randsample(m, k, true, ell);

    % Residuals of the proposals against the rows already chosen.
    C = Q(T,:)';                    % k-by-k, this is Q'(:,T)
    C = C - U*(U'*C);
    C = C - U*(U'*C);               % reorthogonalize
    H = C'*C;

    Tnew = rejection_submatrix(H, ell(T), T);       % Algorithm 4.1
    if isempty(Tnew)
        continue
    end
    room = k - ns;
    if numel(Tnew) > room
        Tnew = Tnew(1:room);
    end

    % Extend the orthonormal basis by the accepted rows.  Anything that came
    % out numerically dependent is dropped rather than admitted to U.
    Z  = Q(Tnew,:)';
    Z  = Z - U*(U'*Z);
    Z  = Z - U*(U'*Z);
    ok = vecnorm(Z) > max(size(Z)) * eps * sqrt(k);
    Z  = Z(:,ok);
    Tnew = Tnew(ok);
    if isempty(Tnew)
        continue
    end

    [Qz, ~] = qr(Z, 0);
    U = [U Qz];                                     %#ok<AGROW> at most k wide

    S(ns+1 : ns+numel(Tnew)) = Tnew;
    ns = ns + numel(Tnew);
end

S = sort(S(1:ns));

end

% -------------------------------------------------------------------------

function Tnew = rejection_submatrix(H, ellT, T)
%REJECTION_SUBMATRIX  Algorithm 4.1: accept or reject a block of proposals.
%   Walks the block in order.  Proposal i is accepted with probability
%   H(i,i)/ellT(i), and on acceptance the remaining diagonal is deflated by a
%   Cholesky elimination step so that later proposals in the same block are
%   tested against their residual after this pivot is inducted.

b    = numel(T);
Tnew = zeros(0,1);

for i = 1:b
    if H(i,i) > 0 && ellT(i)*rand() < H(i,i)
        Tnew(end+1,1) = T(i);                       %#ok<AGROW> at most b long
        H(i:b, i:b) = H(i:b,i:b) - H(i:b,i)*H(i,i:b)/H(i,i);
    end
end

end
