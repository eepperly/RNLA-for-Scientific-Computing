classdef KernelOracle < handle
%KERNELORACLE  Entry access to a Gaussian kernel matrix, with an exact count.
%   K = KERNELORACLE(X,sigma) represents the n-by-n kernel matrix
%
%       A(i,j) = exp(-||x_i - x_j||^2 / (2 sigma^2))
%
%   on the rows x_i of X, without forming it.  Every entry an algorithm reads
%   goes through one of the methods below, which add the number of entries
%   evaluated to K.count.  Entry accesses are therefore counted by
%   construction rather than by a formula for each algorithm, so the count
%   includes whatever an implementation actually evaluates -- rejected
%   proposals, repeated reads, everything.
%
%   Methods:
%       d = K.diagonal()      the diagonal, n entries
%       C = K.cols(I)         A(:,I), n*|I| entries
%       H = K.sub(I)          A(I,I), |I|^2 entries
%       B = K.block(I,J)      A(I,J), |I|*|J| entries
%       K.reset()             zero the counter
%
%   See also GAUSS_KERNEL.

    properties (SetAccess = private)
        X        % n-by-p data
        sqn      % squared row norms of X
        sigma    % bandwidth
        n        % number of points
    end
    properties
        count = 0
    end

    methods
        function obj = KernelOracle(X, sigma)
            obj.X     = X;
            obj.sqn   = sum(X.^2, 2);
            obj.sigma = sigma;
            obj.n     = size(X, 1);
        end

        function reset(obj)
            obj.count = 0;
        end

        function d = diagonal(obj)
            d = ones(obj.n, 1);                 % exp(0) for the Gaussian kernel
            obj.count = obj.count + obj.n;
        end

        function C = cols(obj, I)
            C = obj.block(1:obj.n, I);
        end

        function H = sub(obj, I)
            H = obj.block(I, I);
        end

        function B = block(obj, I, J)
            B = gauss_kernel(obj.X(I,:), obj.X(J,:), obj.sigma, ...
                             obj.sqn(I), obj.sqn(J));
            obj.count = obj.count + numel(B);
        end
    end
end
