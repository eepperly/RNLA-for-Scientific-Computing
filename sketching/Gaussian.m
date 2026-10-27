function [SA, SB] = Gaussian(A, B, k)
%GAUSSIAN  Dense Gaussian sketch of A and B.
%   [SA,SB] = GAUSSIAN(A,B,k) returns Om'*A and Om'*B, where Om is an m-by-k
%   matrix of i.i.d. N(0,1/k) entries and m = size(A,1).  E[Om*Om'] = I, so
%   the sketch preserves norms in expectation.
%
%   B may be [], in which case SB is empty.  A and B must have the same number
%   of rows; the same Om is applied to both.
%
%   This is the reference embedding: the best distortion per row of any sketch
%   here, and the only one whose distribution is rotationally invariant.  It is
%   also by far the most expensive, at O(k*m*n) flops with no fast transform to
%   exploit -- for a tall A it is a baseline to measure the others against, not
%   a practical choice.
%
%   Om is never formed in full.  It is generated and consumed one row block at
%   a time, so memory stays at one block regardless of m; a full Om for
%   m = 10^6 and k = 2000 would be 16 GB.
%
%   See also SPARSESTACK, SRFT, MULTISKETCH, LESS, UNIFORMSAMPLING, LEVERAGESAMPLING.

[m, B] = check_sketch_args(A, B, k, mfilename);

SA = zeros(k, size(A,2));
SB = zeros(k, size(B,2));

% Block so that each chunk of Om holds about 10^7 entries.
blk = max(1, floor(1e7/k));
for lo = 1:blk:m
    hi = min(lo+blk-1, m);
    G  = randn(k, hi-lo+1) / sqrt(k);
    SA = SA + G * A(lo:hi,:);
    SB = SB + G * B(lo:hi,:);
end

end
