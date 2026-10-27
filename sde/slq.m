function [nodes, weights] = slq(A, n, m, t, reorth)
%SLQ  Stochastic Lanczos quadrature for spectral density estimation.
%   [nodes,weights] = SLQ(A,n,m,t) returns a discrete measure
%   muhat = sum_j weights(j) delta_{nodes(j)} approximating the spectral
%   measure mu_A of the n-by-n symmetric matrix A, which places a unit mass on
%   each eigenvalue and so has total mass n.
%
%   m random probes are used, each run for t Lanczos steps.  For probe i the
%   Lanczos tridiagonal T is eigendecomposed, its eigenvalues become quadrature
%   nodes, and the weights are ||w_i||^2 times the squared first components of
%   the eigenvectors.  By the Lanczos-Gaussian quadrature connection this is a
%   t-point Gaussian rule for the local spectral measure mu_{A,w_i}, exact for
%   every polynomial of degree < 2t, so t steps deliver 2t-1 exact moments.
%   Averaging over probes approximates the global measure.
%
%   Probes are drawn uniformly from the sphere of radius sqrt(n), so
%   ||w_i||^2 = n and the returned measure has total mass n.
%
%   [nodes,weights] = SLQ(A,n,m,t,reorth) chooses whether the Lanczos basis is
%   fully reorthogonalized (default true).  Turning it off costs O(n t) rather
%   than O(n t^2) per probe but lets the basis lose orthogonality, which puts
%   ghost nodes into the quadrature rule; see LANCZOS_TRIDIAG.
%
%   A may be an explicit matrix or a function handle computing X -> A*X.
%
%   See also LANCZOS_TRIDIAG, KPM, CHEB_MOMENT_REGRESSION.

if nargin < 5 || isempty(reorth), reorth = true; end

nodes   = [];
weights = [];

for i = 1:m
    w = randn(n,1);
    w = sqrt(n) * w / norm(w);          % uniform on the sphere of radius sqrt(n)

    T = lanczos_tridiag(A, w, t, reorth);
    [U, Lam] = eig(T, 'vector');

    nodes   = [nodes;   Lam];                       %#ok<AGROW>
    weights = [weights; norm(w)^2 * U(1,:)'.^2];    %#ok<AGROW>
end

weights = weights / m;

end
