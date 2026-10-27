function H = tfim(L, J, h)
%TFIM  Transverse-field Ising model Hamiltonian on an open chain.
%   H = TFIM(L) returns the 2^L-by-2^L sparse Hamiltonian
%
%       H = -J * sum_{i=1}^{L-1} sz_i sz_{i+1}  -  h * sum_{i=1}^{L} sx_i
%
%   for L spins with open boundary conditions and J = h = 1, the critical
%   point of the model.  H = TFIM(L,J,h) sets the couplings.
%
%   H is real symmetric with 2^L eigenvalues, and is the standard test case
%   for spectral density estimation: the spectrum is a dense band with sharp
%   edges and internal structure that a low-order polynomial cannot resolve.
%
%   See also SLQ, KPM, CHEB_MOMENT_REGRESSION.

if nargin < 2 || isempty(J), J = 1; end
if nargin < 3 || isempty(h), h = 1; end
validateattributes(L, {'numeric'}, {'scalar','integer','positive','<=',20}, mfilename, 'L', 1);

sx = sparse([0 1; 1 0]);
sz = sparse([1 0; 0 -1]);
N  = 2^L;

H = sparse(N, N);

% Nearest-neighbour zz coupling on sites (i,i+1).
zz = kron(sz, sz);
for i = 1:L-1
    H = H - J * kron(kron(speye(2^(i-1)), zz), speye(2^(L-i-1)));
end

% Transverse field on each site.
for i = 1:L
    H = H - h * kron(kron(speye(2^(i-1)), sx), speye(2^(L-i)));
end

H = (H + H')/2;

end
