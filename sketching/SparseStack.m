function [SA, SB, Omt] = SparseStack(A, B, k, zeta)
%SPARSESTACK  Sparse stack sketch of A and B.
%   [SA,SB] = SPARSESTACK(A,B,k) returns Om'*A and Om'*B, where Om is an
%   m-by-k sparse stack embedding and m = size(A,1).  Om' is built by
%   stacking zeta = 4 independent CountSketch matrices, one per block of its
%   k rows, scaled by 1/sqrt(zeta): every column of Om' has exactly zeta
%   nonzeros, one per block, at a uniformly random position with a random
%   +/-1 sign.  E[Om*Om'] = I, so the sketch preserves norms in expectation.
%
%   [SA,SB] = SPARSESTACK(A,B,k,zeta) uses zeta blocks instead of 4.  zeta is
%   capped at k so every block holds at least one row.
%
%   B may be [], in which case SB is empty.  A and B must have the same
%   number of rows; the same Om is applied to both.
%
%   Cost is zeta nonzero multiplies per entry of A and B.
%
%   See also SRFT, UNIFORMSAMPLING, LEVERAGESAMPLING.

if nargin < 4 || isempty(zeta), zeta = 4; end
[m, B] = check_sketch_args(A, B, k, mfilename);
validateattributes(zeta, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'zeta', 4);

zeta  = min(zeta, k);
edges = round(linspace(0, k, zeta+1));          % row-block boundaries of Om'

rows = zeros(zeta, m);
for b = 1:zeta
    lo = edges(b);
    rows(b,:) = lo + randi(edges(b+1) - lo, 1, m);
end
cols = repmat(1:m, zeta, 1);
vals = (2*randi(2, zeta, m) - 3) / sqrt(zeta);  % random +/-1, scaled

% Row blocks are disjoint and each column contributes one entry per block, so
% there are no duplicate (row,col) pairs for sparse() to sum.
Omt = sparse(rows(:), cols(:), vals(:), k, m);

SA = Omt * A;
SB = Omt * B;

end
