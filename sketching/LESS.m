function [SA, SB, ell] = LESS(A, B, k, s, zeta, kproj)
%LESS  Leverage score sparsified sketch of A and B.
%   [SA,SB] = LESS(A,B,k) returns Om'*A and Om'*B, where Om' is a k-by-m
%   sparse embedding whose nonzero positions are drawn from the estimated
%   leverage score distribution of A.  Each of the k rows independently draws
%   s indices j_1..j_s i.i.d. with P(j) = p_j proportional to the estimated
%   leverage score, and is
%
%       (1/sqrt(k)) * (1/sqrt(s)) * sum_t  x_t / sqrt(p_{j_t}) * e_{j_t}',
%
%   with x_t independent random signs.  Repeated draws within a row add, as
%   the definition requires.  E[Om*Om'] = I, so the sketch preserves norms in
%   expectation.
%
%   [SA,SB,ell] = LESS(...) also returns the estimated leverage scores.
%
%   [...] = LESS(A,B,k,s,zeta,kproj) sets the number of nonzeros per row s
%   (default ceil(log(m))) and passes zeta and kproj to ESTIMATE_LEVERAGE.
%
%   LESS sits between SPARSESTACK and LEVERAGESAMPLING.  Like a sparse stack
%   it has a fixed number of nonzeros per row and mixes rather than selects,
%   so it does not discard the rows it misses; like leverage sampling it puts
%   its mass where the row space actually lives, so it embeds coherent
%   matrices that a uniform sparse sketch would need many more rows to handle.
%
%   The scores come from A alone, not [A B], matching ESTIMATE_LEVERAGE.
%
%   See also ESTIMATE_LEVERAGE, LEVERAGESAMPLING, SPARSESTACK, GAUSSIAN, SRFT.

narginchk(3,6);
if nargin < 4, s     = []; end
if nargin < 5, zeta  = []; end
if nargin < 6, kproj = []; end
[m, B] = check_sketch_args(A, B, k, mfilename);
if isempty(s)
    s = ceil(log(m));
end
validateattributes(s, {'numeric'}, {'scalar','integer','positive'}, mfilename, 's', 4);

ell = estimate_leverage(A, zeta, kproj);

p   = max(ell, 0);
p   = p / sum(p);
idx = reshape(randsample(m, k*s, true, p), k, s);   % k rows, s draws each

rows = repmat((1:k)', 1, s);
sgn  = 2*randi(2, k, s) - 3;
vals = sgn ./ sqrt(s * k * p(idx));

% Duplicate (row,col) pairs are summed by sparse(), which is exactly the sum
% over repeated draws in the LESS definition.
Omt = sparse(rows(:), idx(:), vals(:), k, m);

SA = Omt * A;
SB = Omt * B;

end
