function [Afun, n] = as_operator(A, n)
%AS_OPERATOR  Normalize a matrix or an implicit operator to a function handle.
%   Afun = AS_OPERATOR(A) returns a handle computing X -> A*X.  A may be an
%   explicit matrix (full, sparse, or any class supporting mtimes and size), or
%   already a function handle, in which case it is passed through.
%
%   [Afun,n] = AS_OPERATOR(A,n) also returns the dimension.  For an explicit A
%   this is size(A,1) and the supplied n is ignored; for a function handle
%   there is nothing to query, so n must be given.
%
%   Afun must accept a block of columns, not just a single vector: the Nystrom
%   routines apply it to an n-by-k test matrix in one go.
%
%   See also NYSTROM, NYSTROMSUBSPACE, NYSTROMKRYLOV, NYSTROM_FACTOR.

if isa(A, 'function_handle')
    Afun = A;
    if nargout > 1 && (nargin < 2 || isempty(n))
        error('as_operator:SizeUnknown', ...
              ['When A is a function handle its dimension cannot be queried; ' ...
               'pass n explicitly.']);
    end
else
    Afun = @(X) A*X;
    n    = size(A,1);
end

end
