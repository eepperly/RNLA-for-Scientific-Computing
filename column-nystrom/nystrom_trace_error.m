function e = nystrom_trace_error(C, S, dA)
%NYSTROM_TRACE_ERROR  Relative trace error of a column Nystrom approximation.
%   e = NYSTROM_TRACE_ERROR(C,S,dA) returns
%
%       ( tr(A) - tr(A<S>) ) / tr(A),     A<S> = C * A(S,S)^+ * C',
%
%   from the landmark columns C = A(:,S) and the diagonal dA of A.  Since
%   A - A<S> is psd, this is its trace norm relative to that of A.
%
%   A<S> = F*F' with F = C * W^{+1/2}, W = A(S,S) = C(S,:), and the pseudo-
%   inverse square root taken from an eigendecomposition with the standard
%   numerical-rank cutoff |S|*eps*max(eig(W)).  The cutoff matters only when
%   W is singular, e.g. when two landmarks are identical points (the Adult data
%   contain exact duplicate rows); the pseudoinverse is then the correct
%   definition of A<S>.  The residual diagonal dA - rownorms(F)^2 is clipped
%   at zero against roundoff before summing.
%
%   The same routine is used for every selection method, including those that
%   already hold a Cholesky factor, so all are measured identically.
%
%   See also NYSTROM_KRR.

W = C(S, :);
W = (W + W') / 2;
[V, E] = eig(W);
ev   = diag(E);
keep = ev > numel(S) * eps * max(ev);
F    = C * (V(:, keep) ./ sqrt(ev(keep))');
e    = sum(max(dA - sum(F.^2, 2), 0)) / sum(dA);

end
