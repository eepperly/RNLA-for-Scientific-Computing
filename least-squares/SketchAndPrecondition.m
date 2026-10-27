function [x, X] = SketchAndPrecondition(A, b, iters, init, refine_at, d, zeta)
%SKETCHANDPRECONDITION  Sketch-and-precondition least squares, with optional
%   iterative refinement.
%   x = SKETCHANDPRECONDITION(A,b,iters) approximately solves min ||b - A*x||
%   for tall A by conjugate gradient on the randomly preconditioned normal
%   equations, run for iters iterations from the sketch-and-solve initial guess.
%
%   x = SKETCHANDPRECONDITION(A,b,iters,init) chooses the initialization:
%   'sketch' (default) for sketch-and-solve, or 'zero'.
%
%   x = SKETCHANDPRECONDITION(A,b,iters,init,refine_at) performs one step of
%   iterative refinement after refine_at iterations: the correction so far is
%   folded into x, the residual b - A*x is recomputed, and CG restarts on the new
%   right-hand side for the remaining iters - refine_at iterations.  refine_at =
%   0 (default) means no refinement.  This is the refinement of Epperly, Meier &
%   Nakatsukasa, "Fast randomized least-squares solvers can be just as accurate
%   and stable as classical direct solvers" (CPAM 2025), Algorithm 1/4:
%
%       c   = R'\(A'*(b - A*x))
%       dy  = CG(z -> R'\(A'*(A*(R\z))), c)
%       x   = x + R\dy
%
%   x = SKETCHANDPRECONDITION(A,b,iters,init,refine_at,d,zeta) sets the embedding
%   dimension d (default 12n) and nonzeros per column zeta (default 4).
%
%   [x,X] = SKETCHANDPRECONDITION(...) also returns the n-by-(iters+1) history of
%   iterates, X(:,1) being the initial guess and X(:,j+1) the iterate after j
%   iterations.  Without the second output nothing is recorded, so timed runs
%   pay only for the solve.
%
%   iters = 0 returns the initial guess, i.e. plain sketch-and-solve.
%
%   This is the QR-based method of the paper's Algorithm 4: sketch, S*A = Q*R,
%   sketch-and-solve start x = R\(Q'*(S*b)), and CG on the normal equations
%   preconditioned by R, with R^{-1} applied by triangular solves.  The sketch
%   is a sparse embedding with zeta = 4 and d = 12n (SparseStack, the
%   survey's sparse sign construction).
%
%   The paper's Section 4.2 makes further recommendations not used here.  Its
%   SVD in place of QR is there to support the sketched Karlson-Walden
%   estimate that drives adaptive stopping, and this routine runs a fixed
%   number of iterations instead.  Its column scaling is left to the caller.  With QR there is also no truncated
%   preconditioner, so A must be numerically full rank (cond(A) << 1/u).
%
%   See also SPARSESTACK.

[~, n] = size(A);
if nargin < 4 || isempty(init),      init      = 'sketch'; end
if nargin < 5 || isempty(refine_at), refine_at = 0;        end
if nargin < 6 || isempty(d),         d         = 12*n;     end
if nargin < 7 || isempty(zeta),      zeta      = 4;        end
record = nargout > 1;

% Sketch and QR factorize
[SA, Sb] = SparseStack(A, b, d, zeta);
[Q, R]   = qr(SA, 0);

switch init
    case 'sketch'
        x = R \ (Q'*Sb);                    % sketch-and-solve
    case 'zero'
        x = zeros(n,1);
    otherwise
        error('SketchAndPrecondition:BadInit', ...
              'init must be ''sketch'' or ''zero'', not ''%s''.', init);
end

if record
    X = zeros(n, iters+1);
    X(:,1) = x;
end
if iters == 0
    return
end

% CG on (R'\A'*A/R) dy = R'\A'*(b - A*x)
[dy, res, p, rho] = cg_start(A, b, x, R);
for j = 1:iters
    if refine_at > 0 && j == refine_at + 1
        x = x + R\dy;                       % iterative refinement: fold in the
        [dy, res, p, rho] = cg_start(A, b, x, R);   % correction, restart CG
    end
    if rho > 0
        q     = R' \ (A' * (A * (R\p)));
        alpha = rho / (p'*q);
        dy    = dy + alpha*p;
        res   = res - alpha*q;
        rhon  = res'*res;
        p     = res + (rhon/rho)*p;
        rho   = rhon;
    end
    if record
        X(:,j+1) = x + R\dy;
    end
end
x = x + R\dy;

end

function [dy, res, p, rho] = cg_start(A, b, x, R)
%CG_START  Right-hand side and initial state for a fresh CG solve.
c   = R' \ (A' * (b - A*x));
dy  = zeros(size(R,2), 1);
res = c;
p   = res;
rho = res'*res;
end
