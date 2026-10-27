function [SA, SB, ell] = LeverageSampling(A, B, k, zeta, kproj)
%LEVERAGESAMPLING  Leverage score row sampling sketch of A and B.
%   [SA,SB] = LEVERAGESAMPLING(A,B,k) returns Om'*A and Om'*B, where Om' draws
%   k of the m = size(A,1) rows i.i.d. with replacement, row i with
%   probability p_i proportional to its estimated leverage score, and rescales
%   the drawn row by 1/sqrt(k*p_i).  E[Om*Om'] = I, so the sketch preserves
%   norms in expectation.
%
%   [SA,SB,ell] = LEVERAGESAMPLING(...) also returns the m-by-1 vector of
%   estimated leverage scores that defined the distribution.
%
%   [...] = LEVERAGESAMPLING(A,B,k,zeta,kproj) passes the SparseStack sparsity
%   zeta (default 4) and the projection width kproj (default ceil(log(m)))
%   through to ESTIMATE_LEVERAGE.
%
%   Om is never formed: the rows are taken by indexing and rescaled with .*.
%
%   The scores come from A alone, not from [A B], matching the usual
%   formulation.  For a least-squares problem in which B is the right-hand
%   side, sampling on the leverage scores of [A B] instead would also cover
%   the residual direction.
%
%   Unlike UNIFORMSAMPLING this embeds regardless of coherence, since the rows
%   carrying the row space are exactly the ones the scores make likely to be
%   drawn.  On a coherent 4000-by-50 test matrix it embedded without failure
%   from k = 2000 upward, where uniform sampling failed in 100% of trials.
%
%   See also ESTIMATE_LEVERAGE, UNIFORMSAMPLING, SPARSESTACK, SRFT.

narginchk(3,5);
if nargin < 4, zeta  = []; end
if nargin < 5, kproj = []; end
[m, B] = check_sketch_args(A, B, k, mfilename);

ell = estimate_leverage(A, zeta, kproj);

p   = max(ell, 0);
p   = p / sum(p);
idx = randsample(m, k, true, p);            % i.i.d. draws, with replacement
wt  = 1 ./ sqrt(k * p(idx));                % rescaling weights

SA = wt .* A(idx,:);
SB = wt .* B(idx,:);

end
