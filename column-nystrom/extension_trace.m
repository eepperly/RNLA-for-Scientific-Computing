function t = extension_trace(rows, S)
%EXTENSION_TRACE  Trace of the column Nystrom approximation from sampled rows.
%   t = EXTENSION_TRACE(rows,S) returns tr(A<S>) given rows = A(S,:), as the
%   reference code computes it for index sets from uniform and RLS sampling
%   (NystromExtension in lra.py):
%
%       L = chol(W + tr(W)*eps*I),   W = A(S,S),   G = L \ rows,   t = ||G||_F^2.
%
%   The shift tr(W)*eps makes the Cholesky factorization succeed when W is
%   singular to working precision (e.g. repeated points).  If MATLAB's chol
%   still finds W + shift indefinite where numpy's did not, the shift is
%   raised by factors of 10 until it succeeds, and a warning is issued.
%
%   See also PIVOTED_CHOLESKY.

W = rows(:, S);
W = (W + W') / 2;
shift = trace(W) * eps;
[L, flag] = chol(W + shift * eye(numel(S)), 'lower');
bumped = false;
while flag ~= 0
    shift = 10 * shift;
    bumped = true;
    [L, flag] = chol(W + shift * eye(numel(S)), 'lower');
end
if bumped
    warning('extension_trace:Shift', ...
            'Shift raised to %.2e (tr(W)*eps = %.2e) to factor W.', shift, trace(W)*eps);
end
G = L \ rows;
t = norm(G, 'fro')^2;

end
