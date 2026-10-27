function [SA, SB] = UniformSampling(A, B, k)
%UNIFORMSAMPLING  Uniform row sampling sketch of A and B.
%   [SA,SB] = UNIFORMSAMPLING(A,B,k) returns Om'*A and Om'*B, where Om' draws
%   k of the m = size(A,1) rows uniformly at random, with replacement, and
%   rescales each by sqrt(m/k).  E[Om*Om'] = I, so the sketch preserves norms
%   in expectation.
%
%   Om is never formed: the rows are taken by indexing and rescaled with .*,
%   so the cost is a copy of k rows rather than a matrix product.
%
%   B may be [], in which case SB is empty.  A and B must have the same number
%   of rows; the same rows are drawn from both.
%
%   Uniform sampling is a subspace embedding only when the row space of A is
%   incoherent.  When it may not be, use LEVERAGESAMPLING: on a coherent test
%   matrix uniform sampling failed to embed in 100% of trials even at k = m,
%   because sampling with replacement still misses many rows.
%
%   See also LEVERAGESAMPLING, SPARSESTACK, SRFT.

[m, B] = check_sketch_args(A, B, k, mfilename);

idx = randsample(m, k, 1);                   % i.i.d. uniform, with replacement

SA = sqrt(m/k) * A(idx,:);
SB = sqrt(m/k) * B(idx,:);

end
