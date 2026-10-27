function Y = lanczos_fun(H, f, X, k)
%LANCZOS_FUN  Matrix-function-times-vectors f(H)*X by k steps of Lanczos.
%   Y = LANCZOS_FUN(H,f,X,k) approximates f(H)*X for a symmetric H, where f is
%   a function handle applied elementwise to eigenvalues.  Each column x of X
%   gets its own Lanczos process, run for exactly k steps: with orthonormal
%   basis V_k and tridiagonal T_k, the approximation is
%
%       f(H) x  ~  ||x|| * V_k * f(T_k) * e_1.
%
%   The default is k = 20.  The columns are advanced in lockstep, so each step
%   is one sparse-times-block product H*V rather than b separate matvecs, and
%   one product with A = f(H) costs exactly k products with H.
%
%   The step count is fixed rather than adaptive, so every estimator pays the
%   same price per matvec with A and is fed the same quality of matvec.  The
%   price is that the result is only as accurate as a degree-(k-1) polynomial
%   approximation of f on the spectrum of H allows, and it is not refined
%   further.  Note also that x -> ||x|| V_k f(T_k) e_1 is not a linear map --
%   V_k depends on x -- so the trace estimators see an approximate,
%   slightly nonlinear operator, and their error includes this Lanczos error on
%   top of their own sampling error.
%
%   If the Krylov space becomes invariant before step k, that column is exact
%   and is frozen there.
%
%   No reorthogonalization is done.  For f(H)b, unlike for eigenvalues, the
%   approximation stays accurate in floating point despite loss of
%   orthogonality (Druskin-Knizhnerman, Greenbaum), and skipping it keeps the
%   cost at O(nnz(H)) per step per column.
%
%   See also TRACE_EXPERIMENT.

if nargin < 4 || isempty(k)
    k = 20;
end
validateattributes(k, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'k', 4);

[n, b] = size(X);
Y = zeros(n, b);

% Process columns in chunks so the stored bases (n x (k+1) x chunk) stay small.
chunk = 32;
for c0 = 1:chunk:b
    idx = c0:min(c0+chunk-1, b);
    Y(:,idx) = lanczos_chunk(H, f, X(:,idx), k);
end

end

% -------------------------------------------------------------------------

function Y = lanczos_chunk(H, f, X, k)

[n, b] = size(X);
nrm   = vecnorm(X);
V     = zeros(n, k, b);
alpha = zeros(k, b);
beta  = zeros(k, b);

Vj   = X ./ nrm;
Vjm1 = zeros(n, b);

for j = 1:k
    V(:,j,:) = reshape(Vj, n, 1, b);

    W = H*Vj;
    alpha(j,:) = sum(Vj.*W, 1);
    W = W - alpha(j,:).*Vj;
    if j > 1
        W = W - beta(j-1,:).*Vjm1;
    end
    if j == k
        break
    end
    beta(j,:) = vecnorm(W);

    % Invariant subspace reached: the approximation is already exact for that
    % column.  Zero the coupling and feed zeros forward, so the remaining
    % steps add a decoupled block to T that has no weight on e_1.
    dead = beta(j,:) <= eps * max(1, abs(alpha(j,:)));
    beta(j,dead) = 0;
    W(:,dead) = 0;
    bj = beta(j,:);  bj(dead) = 1;

    Vjm1 = Vj;
    Vj   = W ./ bj;
end

% ||x|| * V_k * f(T_k) * e_1 for each column.
Y = zeros(n, b);
for c = 1:b
    T = diag(alpha(:,c)) + diag(beta(1:k-1,c),1) + diag(beta(1:k-1,c),-1);
    [U, D] = eig(T, 'vector');
    fe1 = U * (f(D) .* U(1,:)');
    Y(:,c) = nrm(c) * (V(:,:,c) * fe1);
end

end
