classdef MatrixOracle < handle
%MATRIXORACLE  Entry access to an explicitly stored psd matrix.
%   K = MATRIXORACLE(A) wraps a dense matrix in the same interface as
%   KERNELORACLE (n, diagonal, cols, sub, block, count, reset), so the
%   selection routines written against that interface run unchanged on a
%   precomputed matrix.  Entry accesses are counted the same way.
%
%   See also KERNELORACLE.

    properties (SetAccess = private)
        A
        n
    end
    properties
        count = 0
    end

    methods
        function obj = MatrixOracle(A)
            obj.A = A;
            obj.n = size(A, 1);
        end
        function reset(obj),           obj.count = 0;                          end
        function d = diagonal(obj),    d = diag(obj.A);  obj.count = obj.count + obj.n; end
        function C = cols(obj, I),     C = obj.block(1:obj.n, I);              end
        function H = sub(obj, I),      H = obj.block(I, I);                    end
        function B = block(obj, I, J)
            B = obj.A(I, J);
            obj.count = obj.count + numel(B);
        end
    end
end
