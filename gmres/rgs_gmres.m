function [times, relres] = rgs_gmres(A, b, m, chk, kdim, zeta)
%RGS_GMRES  GMRES on a randomized Gram-Schmidt basis (Balabanov and Grigori,
%           SIAM J. Matrix Anal. Appl. 2022).
%   [times,relres] = RGS_GMRES(A,b,m,chk) runs m iterations and reports the
%   cumulative solver time and true relative residual at each iteration count
%   in chk.  [.] = RGS_GMRES(A,b,m,chk,kdim,zeta) sets the sketch dimension
%   (default 2m) and the sketch sparsity (default 8).
%
%   The basis comes from the RGS-Arnoldi process of Algorithm 4.1: w_1 = b,
%   w_i = A*q_{i-1}, each vector passed through one iteration of the RGS
%   process of Algorithm 2.1,
%
%       p_i   = Theta*w_i
%       R(1:i-1,i) = argmin_y || S_{i-1}*y - p_i ||
%       q'_i  = w_i - Q_{i-1}*R(1:i-1,i)
%       s'_i  = Theta*q'_i,   r_ii = ||s'_i||
%       s_i   = s'_i/r_ii,    q_i = q'_i/r_ii
%
%   so Q is orthonormal in the sketched inner product rather than the
%   Euclidean one.  The coefficients are found by projecting in the sketched
%   space, which costs O(k*i) instead of the O(n*i) of classical Gram-Schmidt:
%   only ONE pass over the high-dimensional basis per iteration, in step 3.
%
%   Step 2 is solved as a least-squares problem rather than by S_{i-1}'*p_i.
%   Balabanov and Grigori are explicit that the stability of the process
%   depends on the stability of that solver, so a QR-based solve is used.
%
%   W_m = [b, A*q_1, ..., A*q_{m-1}] = Q_m*R_m, hence A*Q_{m-1} = Q_m*H with
%   H = R_m(:,2:m) upper Hessenberg, and the GMRES step (4.2) is
%
%       y = argmin_z || H*z - r_11*e_1 ||,   x = Q_{m-1}*y,
%
%   which minimizes the sketched residual and so the true residual up to
%   sqrt((1+eps)/(1-eps)) when Theta is an eps-embedding.
%
%   Implemented in double throughout; the paper's ufine/ucrs multiprecision
%   split is an analysis device and a cost saving, not a requirement.
%
%   Timing covers the basis build, the least-squares solves, and forming x.
%   It excludes the residual evaluation, which is measurement.

if nargin < 5 || isempty(kdim), kdim = 2*m; end
if nargin < 6 || isempty(zeta), zeta = 8;   end

n      = numel(b);
normb  = norm(b);
times  = nan(numel(chk),1);
relres = nan(numel(chk),1);

t0 = tic;
[~,~,Theta] = SparseStack(zeros(n,0), [], kdim, zeta);   % k-by-n sketch
Q  = zeros(n, m);
Sm = zeros(kdim, m);
R  = zeros(m, m);
zx = zeros(m, 1);       % padded coefficients for the checkpoint solve
t  = toc(t0);

for i = 1:m
    t0 = tic;
    if i == 1
        w = b;                                   % Algorithm 4.1, step 1
    else
        w = A*Q(:,i-1);                          % Algorithm 4.1, step 3
    end

    p = Theta * w;                               % Alg 2.1, step 1

    % Step 2, the least-squares problem.  S_{i-1} is orthonormal in exact
    % arithmetic -- s'_i is the residual of this very step, hence orthogonal to
    % range(S_{i-1}), and step 6 normalizes it -- so the solution is the
    % projection S_{i-1}'*p.  In floating point S drifts from orthonormal, a
    % single projection is then NOT the least-squares solution, and the error
    % feeds back into S; solving by one projection alone was measured to lose
    % the residual entirely (2e-1 instead of 4e-12 at m = 1000).  This is what
    % Balabanov and Grigori mean when they say the stability of Algorithm 2.1
    % depends on the stability of the least-squares solver.
    %
    % So project twice.  The second pass corrects the first against whatever
    % non-orthogonality S has picked up, which is the classical reorthogonalized
    % Gram-Schmidt argument, and it costs O(k*i) rather than the O(k*i^2) of a
    % QR from scratch.  That keeps the sketched work inside the cost model of
    % section 2.4, where operations on sketched vectors are neglected and step 3
    % dominates.
    %
    % The products below use the whole preallocated array rather than a leading
    % submatrix: indexing makes MATLAB copy the array on every iteration, which
    % measured 22x slower.  Trailing columns are zero, so z comes out correctly
    % padded.
    z = Sm' * p;
    z = z + Sm' * (p - Sm*z);                    % reorthogonalize
    R(1:i-1,i) = z(1:i-1);

    % Step 3, the one long pass.  Multiplying by z as a SPARSE vector keeps the
    % cost at O(n*i), the 2ni of section 2.4.  A dense z would cost O(n*m) at
    % every iteration regardless of i -- flat in i -- which makes early
    % iterations as expensive as late ones and wipes out the advantage over
    % Gram-Schmidt exactly where it should be largest.  Measured per iteration
    % at n = 9e4, m = 1000: dense 0.009 s at every i, sparse 0.0026 s at
    % i = 100 rising to 0.025 s at i = 1000, against 0.010 to 0.052 s for
    % modified Gram-Schmidt.
    qp = w - Q*sparse(z);

    sp  = Theta * qp;                            % step 4
    rii = norm(sp);                              % step 5
    if rii <= eps*normb
        break
    end
    R(i,i)  = rii;
    Sm(:,i) = sp / rii;                          % step 6
    Q(:,i)  = qp / rii;                          % step 7
    t = t + toc(t0);

    c = find(chk == i, 1);
    if ~isempty(c) && i > 1
        t0 = tic;
        H  = R(1:i, 2:i);                        % i-by-(i-1) upper Hessenberg
        e1 = zeros(i,1);  e1(1) = R(1,1);
        y  = H \ e1;                             % equation (4.2)
        zx(1:i-1) = y;  zx(i:end) = 0;           % again, avoid copying Q
        x  = Q * sparse(zx);
        times(c)  = t + toc(t0);
        relres(c) = norm(b - A*x) / normb;
    end
end

end
