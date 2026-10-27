function [SA, SB] = Multisketch(A, B, k, k1)
%MULTISKETCH  CountSketch followed by a Gaussian sketch.
%   [SA,SB] = MULTISKETCH(A,B,k) returns Om'*A and Om'*B for the composition
%   Om' = G' * C', where C' is a CountSketch taking m = size(A,1) rows down to
%   k1 = 12*n^2 (n = size(A,2)), and G' is a dense Gaussian taking k1 down to k.
%   Both stages are norm preserving in expectation, so E[Om*Om'] = I.
%
%   [SA,SB] = MULTISKETCH(A,B,k,k1) sets the intermediate dimension.  k1 is
%   clamped to [k, m]: it cannot exceed the number of rows, and reducing below
%   k would throw away the target dimension.
%
%   The point of the composition is cost.  A Gaussian sketch applied straight
%   to A costs O(k*m*n); the CountSketch stage costs O(nnz(A)) and leaves an
%   12n^2-by-n matrix, after which the Gaussian stage costs only O(k*n^3).  The
%   composition inherits close to Gaussian distortion at a fraction of the
%   price, provided k1 is large enough for the CountSketch to embed.
%
%   B may be [], in which case SB is empty.  A and B must have the same number
%   of rows; the same Om is applied to both.
%
%   See also SPARSESTACK, GAUSSIAN, SRFT, LESS, UNIFORMSAMPLING, LEVERAGESAMPLING.

[m, B] = check_sketch_args(A, B, k, mfilename);

if nargin < 4 || isempty(k1)
    k1 = 12*size(A,2)^2;
end
validateattributes(k1, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'k1', 4);
k1 = min(max(k1, k), m);

[CA, CB] = SparseStack(A, B, k1, 1);    % CountSketch is a sparse stack, zeta = 1
[SA, SB] = Gaussian(CA, CB, k);

end
