function B = gauss_kernel(Xa, Xb, sigma, sqa, sqb)
%GAUSS_KERNEL  Gaussian kernel block exp(-||a - b||^2 / (2 sigma^2)).
%   B = GAUSS_KERNEL(Xa,Xb,sigma) returns the size(Xa,1)-by-size(Xb,1) block.
%   GAUSS_KERNEL(Xa,Xb,sigma,sqa,sqb) reuses precomputed squared row norms.
%   Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a'b, clipped at zero against
%   roundoff, so a block costs one matrix product.

if nargin < 4, sqa = sum(Xa.^2, 2); end
if nargin < 5, sqb = sum(Xb.^2, 2); end
D2 = max(sqa + sqb' - 2*(Xa*Xb'), 0);
B  = exp(-D2 / (2*sigma^2));

end
