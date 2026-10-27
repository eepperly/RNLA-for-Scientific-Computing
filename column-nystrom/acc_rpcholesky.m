function [S, C, F] = acc_rpcholesky(K, k, b)
%ACC_RPCHOLESKY  Column Nystrom landmarks by accelerated randomly pivoted Cholesky.
%   [S,C,F] = ACC_RPCHOLESKY(K,k) runs accelerated RPCholesky (Epperly, Tropp
%   and Webber, "Embrace rejection", SIMAX 2025), which produces the same
%   random distribution of pivots as RPCholesky but a block
%   at a time.  Returns the pivots S, the columns C = A(:,S), and the factor F
%   with A<S> = F*F'.  K is a KERNELORACLE.
%
%   Each round proposes b indices i.i.d. from the current residual diagonal,
%   forms the b-by-b residual submatrix, and walks it accepting proposal j with
%   probability H(j,j)/u(j), where u is the residual diagonal on entry and H is
%   Schur-complemented by each acceptance; the accepted columns are then added
%   to the factor in one block.
%
%   BLOCK SIZE.  [S,C,F] = ACC_RPCHOLESKY(K,k,b) fixes the block size.  The
%   default, b = 'auto', is the default of the reference implementation
%   (accelerated_rpcholesky in the authors' Python package): start at
%   ceil(k/10) and after each round rescale b so that the rejection step costs
%   about a quarter of the processing step, as measured by wall-clock time,
%   with b kept within [max(ceil(b/3),10), min(ceil(1.5b), ceil(k/3))].
%   The block size changes the cost, never the distribution of the pivots.
%
%   Like the reference, it stops early once the residual trace falls below
%   1e-13 times the original trace.
%
%   Entry accesses: n for the diagonal, b^2 per round for the proposed
%   submatrices (including rejected proposals), and n per accepted pivot.
%
%   Structure follows acc_rpcholesky and rejection_sample_submatrix in the
%   thesis code; the auto block size follows the Python package.
%
%   See also GREEDY_CHOLESKY, UNIFORM_NYSTROM, RLS_NYSTROM.

if nargin < 3 || isempty(b), b = 'auto'; end
auto_b = ischar(b) || isstring(b);
if auto_b
    b = ceil(k/10);
end
stoptol = 1e-13;

n    = K.n;
d    = K.diagonal();
orig = sum(d);
F    = zeros(n, k);
C    = zeros(n, k);
S    = zeros(k, 1);
i    = 0;

while i < k
    if sum(d) <= 0, break, end

    Sp = datasample(1:n, b, 'Weights', d, 'Replace', true);    % i.i.d. proposals

    t0 = tic;
    H  = K.sub(Sp) - F(Sp,1:i) * F(Sp,1:i)';    % residual submatrix
    [T, L] = rejection_sample_submatrix(H, diag(H), k - i);
    trej = toc(t0);

    t0 = tic;
    T = Sp(T);
    l = numel(T);
    if l > 0
        Cn = K.cols(T);
        G  = Cn - F(:,1:i) * F(T,1:i)';          % residual columns
        F(:,i+1:i+l) = G / L';
        C(:,i+1:i+l) = Cn;
        S(i+1:i+l)   = T;
        d = max(d - sum(F(:,i+1:i+l).^2, 2), 0);
        i = i + l;
    end
    tproc = toc(t0);

    if auto_b
        target = ceil(b * tproc / (4 * trej));
        b = max([min([target, ceil(1.5*b), ceil(k/3)]), ceil(b/3), 10]);
    end

    if sum(d) <= stoptol * orig, break, end
end

S = S(1:i);  C = C(:,1:i);  F = F(:,1:i);

end

% -------------------------------------------------------------------------

function [T, L] = rejection_sample_submatrix(H, u, lmax)
%REJECTION_SAMPLE_SUBMATRIX  Accept/reject a block of proposals.
%   Proposal j is accepted with probability H(j,j)/u(j), with H overwritten by
%   its Cholesky factor as pivots are accepted, so later proposals are tested
%   against their residual after the earlier acceptances.  At most lmax are
%   accepted.  L = chol(H_in(T,T), 'lower').  The strict inequality rejects
%   anything with zero residual, including repeats of an accepted index.

b = size(H, 1);
T = zeros(b, 1);
na = 0;
for j = 1:b
    if ~(u(j) * rand() < H(j,j))
        continue
    end
    na = na + 1;
    T(na) = j;
    H(j:b, j) = H(j:b, j) / sqrt(H(j,j));
    H(j+1:b, j+1:b) = H(j+1:b, j+1:b) - H(j+1:b, j) * H(j+1:b, j)';
    if na == lmax, break, end
end
T = T(1:na);
L = tril(H(T, T));

end
