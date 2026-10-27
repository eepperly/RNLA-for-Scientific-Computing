function [times, relres, condT] = sgmres(A, b, m, chk, p, zeta)
%SGMRES  Sketched GMRES (Nakatsukasa and Tropp, SIAM Review 2024).
%   [times,relres] = SGMRES(A,b,m,chk) runs m iterations and reports the
%   cumulative solver time and true relative residual at each iteration count
%   in chk.  [.] = SGMRES(A,b,m,chk,p,zeta) sets the truncation depth
%   (default 2) and the sketch sparsity (default 8).
%
%   [times,relres,condT] = SGMRES(...) also returns condest(T) at each
%   checkpoint, where S*A*B = U*T is the whitening QR below.  This is the
%   condition number diagnostic of Fact 2.3, equation (2.5): T brackets the
%   condition number of the reduced matrix,
%
%       (1-eps)/(1+eps) * cond(T) <= cond(A*B) <= (1+eps)/(1-eps) * cond(T),
%
%   so it is a running, essentially free read on how far the truncated basis
%   has degraded -- the quantity that decides whether the sketched solve is
%   still trustworthy.  It is computed outside the timers, as a diagnostic.
%
%   NOTE ON NOTATION: p is the number of steps of partial orthogonalization,
%   matching the survey.  It is not the s of this file, which follows
%   Nakatsukasa and Tropp in reserving s for the sketch embedding dimension
%   s = 2(m+1).
%
%   The basis B spans the Krylov space but is NOT orthogonalized: it comes
%   from the p-truncated Arnoldi process of (1.9), each new vector being
%   orthogonalized against only the previous p vectors.  The reduced
%   matrix A*B falls out as a by-product.  sGMRES then replaces the GMRES
%   least-squares problem (1.6) by its sketched counterpart (1.7),
%
%       minimize_y || S*(A*B*y - b) ||,
%
%   with an s-by-n sketch and s = 2(m+1).  Only the s-by-j sketch S*(A*B) and
%   the s-vector S*b are ever needed for the solve, so the least-squares
%   problem is tiny no matter how large n is.
%
%   The solve goes through the whitening of Fact 2.3: QR-factor the sketched
%   reduced matrix S*A*B = U*T and use T, rather than forming normal equations
%   or pseudo-inverting.  This is what makes the method robust to a poorly
%   conditioned A*B -- it suffices that cond(A*B) is below about 1/eps.
%
%   Timing covers the basis build, the sketch, the least-squares solve, and
%   forming x.  It excludes the residual evaluation, which is measurement.

if nargin < 5 || isempty(p), p = 2; end
if nargin < 6 || isempty(zeta),   zeta   = 8; end

n      = numel(b);
normb  = norm(b);
s      = 2*(m+1);                        % embedding dimension, section 1.2.1
times  = nan(numel(chk),1);
relres = nan(numel(chk),1);
condT  = nan(numel(chk),1);

t0 = tic;
[~,~,S] = SparseStack(zeros(n,0), [], s, zeta);   % s-by-n sparse embedding
B   = zeros(n,m);
SAB = zeros(s,m);
z   = zeros(m,1);
B(:,1) = b / normb;
Sb  = S * b;
t = toc(t0);

for j = 1:m
    t0 = tic;
    w        = A*B(:,j);
    SAB(:,j) = S*w;                      % sketch the new column of A*B
    if j < m
        for i = max(1,j-p+1):j            % truncated orthogonalization
            w = w - (B(:,i)'*w) * B(:,i);
        end
        nw = norm(w);
        if nw <= eps*normb
            break
        end
        B(:,j+1) = w / nw;
    end
    t = t + toc(t0);

    c = find(chk == j, 1);
    if ~isempty(c)
        t0 = tic;
        [U,T] = qr(SAB(:,1:j), 0);       % whitening, Fact 2.3
        y = T \ (U' * Sb);
        z(1:j) = y;  z(j+1:end) = 0;         % avoid copying the basis
        x = B * z;
        times(c)  = t + toc(t0);
        relres(c) = norm(b - A*x) / normb;
        condT(c)  = condest(T);          % diagnostic, deliberately untimed
    end
end

end
