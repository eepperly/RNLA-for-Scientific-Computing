function [A,b,x,r,S,V] = random_ls_problem(m,n,cond_A,res_size,varargin)
%RANDOM_LS_PROBLEM  Random least-squares problem with known solution.
%   [A,b,x,r,S,V] = RANDOM_LS_PROBLEM(m,n,cond_A,res_size) returns an m-by-n
%   matrix A = U*S*V' with U and V Haar-random with orthonormal columns and
%   singular values logarithmically spaced from 1/cond_A to 1, a random unit
%   vector x, and b = A*x + r with r orthogonal to range(A) and
%   ||r|| = res_size, so x is the least-squares solution.  S and V are the
%   singular values (as a diagonal matrix) and right singular vectors of A.
%
%   [...] = RANDOM_LS_PROBLEM(...,U) uses the given m-by-(n+1) matrix U with
%   orthonormal columns in place of a random one; its last column is the
%   residual direction.
if ~isempty(varargin)
    U = varargin{1};
else
    U = haarorth(m,n+1);
end
V = haarorth(n,n);
S = diag(logspace(-log10(cond_A),0,n));
A = U(:,1:n)*S*V';
x = randn(n,1);
x = x / norm(x);
r = U(:,end) * res_size;
b = A*x + r;
end

function Q = haarorth(m,n)
%HAARORTH  Random m-by-n matrix with orthonormal columns, Haar distributed.
%   The sign fix makes the distribution exactly Haar rather than depending on
%   the sign convention of qr.
[Q,R] = qr(randn(m,n),0);
Q = Q .* sign(diag(R))';
end
