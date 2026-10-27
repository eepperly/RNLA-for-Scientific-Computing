function [SA, SB] = SRFT(A, B, k, rounds)
%SRFT  Subsampled randomized trigonometric transform sketch of A and B.
%   [SA,SB] = SRFT(A,B,k) returns Om'*A and Om'*B, where Om is an m-by-k SRFT
%   built from the DCT and m = size(A,1):
%
%       Om' = sqrt(m/k) * R * (C*D_r) * ... * (C*D_1),
%
%   with C the orthonormal DCT-II, each D_i an independent random +/-1
%   diagonal, and R a uniformly random selection of k of the m rows.
%   E[Om*Om'] = I, so the sketch preserves norms in expectation.
%
%   [SA,SB] = SRFT(A,B,k,rounds) uses `rounds` rounds of randomization
%   (default 2).  Each round is one sign flip followed by one DCT.  On a
%   coherent input a single round leaves a few-percent failure tail that a
%   second round closes; a third buys nothing.
%
%   B may be [], in which case SB is empty.  A and B must have the same number
%   of rows; they are transformed together so both see the same Om.
%
%   Om is never formed.  Cost is rounds*O(m*log(m)) per column, and the row
%   subsampling is done by indexing rather than by a matrix product.  The DCT
%   is MATLAB's dct, so the Signal Processing Toolbox is required.
%
%   See also SPARSESTACK, UNIFORMSAMPLING, LEVERAGESAMPLING.

if nargin < 4 || isempty(rounds), rounds = 2; end
[m, B] = check_sketch_args(A, B, k, mfilename);
validateattributes(rounds, {'numeric'}, {'scalar','integer','nonnegative'}, mfilename, 'rounds', 4);

nA = size(A,2);
C  = [A B];                                     % one Om for both blocks
for r = 1:rounds
    C = dct(C .* (2*randi(2,m,1) - 3));     % random sign flip, then DCT
end

idx = randperm(m, k);                           % subsample by indexing
C   = sqrt(m/k) .* C(idx,:);

SA = C(:, 1:nA);
SB = C(:, nA+1:end);

end