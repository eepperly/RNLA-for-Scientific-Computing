function [S, C] = uniform_nystrom(K, k)
%UNIFORM_NYSTROM  Column Nystrom landmarks by uniform sampling.
%   [S,C] = UNIFORM_NYSTROM(K,k) draws k distinct indices uniformly at random,
%   without replacement,
%   and returns them with the columns C = A(:,S) that define the column Nystrom
%   approximation A<S> = C * A(S,S)^+ * C'.  K is a KERNELORACLE.
%
%   Entry accesses: n*k, for the columns.  The diagonal is not needed.
%
%   See also RLS_NYSTROM, GREEDY_CHOLESKY, ACC_RPCHOLESKY.

S = randsample(K.n, k, false);
C = K.cols(S);

end
