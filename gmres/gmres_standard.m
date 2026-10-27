function [times, relres] = gmres_standard(A, b, m, chk)
%GMRES_STANDARD  GMRES with full Arnoldi (modified Gram-Schmidt).
%   [times,relres] = GMRES_STANDARD(A,b,m,chk) runs m iterations and reports,
%   at each iteration count in chk, the cumulative solver time and the true
%   relative residual ||b - A*x||/||b||.
%
%   The reference method: the basis is fully orthogonalized, so it costs
%   O(n*j) work and one pass over the whole basis at iteration j, and the
%   basis must be stored in full.  Both are what the other three methods try
%   to avoid.
%
%   Timing covers the Arnoldi sweep, the Hessenberg least-squares solve, and
%   forming x.  It excludes the residual evaluation, which is measurement.

n      = numel(b);
normb  = norm(b);
Q      = zeros(n, m+1);
H      = zeros(m+1, m);
z      = zeros(m+1, 1);
times  = nan(numel(chk),1);
relres = nan(numel(chk),1);

t0 = tic;
beta   = normb;
Q(:,1) = b / beta;
t = toc(t0);

for j = 1:m
    t0 = tic;
    w = A*Q(:,j);
    for i = 1:j                          % modified Gram-Schmidt
        H(i,j) = Q(:,i)' * w;
        w = w - H(i,j)*Q(:,i);
    end
    H(j+1,j) = norm(w);
    if H(j+1,j) > eps*normb
        Q(:,j+1) = w / H(j+1,j);
    end
    t = t + toc(t0);

    c = find(chk == j, 1);
    if ~isempty(c)
        t0 = tic;
        e1 = zeros(j+1,1);  e1(1) = beta;
        y  = H(1:j+1,1:j) \ e1;
        z(1:j) = y;  z(j+1:end) = 0;         % avoid copying the basis
        x  = Q * z;
        times(c)  = t + toc(t0);
        relres(c) = norm(b - A*x) / normb;
    end

    if H(j+1,j) <= eps*normb
        break                            % happy breakdown
    end
end

end
