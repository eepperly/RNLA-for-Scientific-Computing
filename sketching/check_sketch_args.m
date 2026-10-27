function [m, B] = check_sketch_args(A, B, k, caller)
%CHECK_SKETCH_ARGS  Shared argument validation for the sketching routines.
%   [m,B] = CHECK_SKETCH_ARGS(A,B,k,caller) checks that A and B have matching
%   row counts and that k is a positive integer no larger than m = size(A,1).
%   An empty B is normalized to an m-by-0 matrix so callers can concatenate
%   and multiply it without special cases.

validateattributes(A, {'numeric'}, {'2d'}, caller, 'A', 1);
validateattributes(k, {'numeric'}, {'scalar','integer','positive'}, caller, 'k', 3);

m = size(A,1);
if isempty(B)
    B = zeros(m, 0);
else
    validateattributes(B, {'numeric'}, {'2d'}, caller, 'B', 2);
    if size(B,1) ~= m
        error([caller ':RowMismatch'], ...
              'A has %d rows but B has %d; they must match.', m, size(B,1));
    end
end

if k > m
    error([caller ':SketchTooLarge'], ...
          'Sketching dimension k = %d exceeds the number of rows m = %d.', k, m);
end

end
