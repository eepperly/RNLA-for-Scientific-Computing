function [lam, eps1] = tfim_eigenvalues(L, J, h)
%TFIM_EIGENVALUES  All 2^L eigenvalues of the TFIM chain, in closed form.
%   lam = TFIM_EIGENVALUES(L) returns the sorted 2^L eigenvalues of the
%   transverse-field Ising Hamiltonian built by TFIM(L), without ever forming
%   the 2^L-by-2^L matrix.  TFIM_EIGENVALUES(L,J,h) sets the couplings.
%
%   [lam,eps1] = TFIM_EIGENVALUES(...) also returns the L single-particle
%   energies.
%
%   The open-chain TFIM is a free-fermion model: the Jordan-Wigner transform
%   turns it into L independent fermionic modes, so
%
%       H = sum_{k=1}^L eps_k (n_k - 1/2),      n_k in {0,1},
%
%   and every eigenvalue is a signed sum  sum_k (+-) eps_k/2  over the 2^L
%   occupation patterns.  For the open chain the single-particle energies are
%   eps_k = 2*sigma_k, where sigma_k are the singular values of the L-by-L
%   bidiagonal matrix with h on the diagonal and J on the subdiagonal.  So the
%   spectrum is every signed sum of those L singular values.
%
%   Cost is an L-by-L SVD plus the 2^L sums, against O(8^L) for a dense
%   eigendecomposition: L = 12 goes from seconds to microseconds, and L = 20 is
%   reachable where forming the matrix at all would not be.
%
%   Verified against eig(full(tfim(L,J,h))) for L = 2,4,6,8,10 and several
%   (J,h), agreeing to machine precision.
%
%   See also TFIM.

if nargin < 2 || isempty(J), J = 1; end
if nargin < 3 || isempty(h), h = 1; end
validateattributes(L, {'numeric'}, {'scalar','integer','positive','<=',26}, mfilename, 'L', 1);

% Single-particle energies: singular values of the bidiagonal coupling matrix.
M     = diag(h*ones(L,1)) + diag(J*ones(L-1,1), -1);
sigma = svd(M);
eps1  = 2*sigma;

% Every signed sum, built by doubling: 2^L values without an explicit loop
% over occupation patterns.
lam = 0;
for k = 1:L
    lam = [lam + sigma(k); lam - sigma(k)];     % doubles L times
end

lam = sort(lam);

end
