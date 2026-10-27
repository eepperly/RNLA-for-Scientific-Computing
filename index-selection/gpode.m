function p = gpode(U, m)
%GPODE  GappyPOD+E sampling point selection.
%   p = GPODE(U,m) returns m sampling points for the N-by-n basis U with
%   orthonormal columns.  The first n points are the QDEIM pivots; each
%   subsequent point is the one that most increases the smallest singular value
%   of U(p,:), estimated by a rank-one update formula.
%
%   This is Algorithm 1 of
%
%     B. Peherstorfer, Z. Drmac and S. Gugercin, "Stability of discrete
%     empirical interpolation and gappy proper orthogonal decomposition with
%     randomized and deterministic sampling points", SIAM J. Sci. Comput.
%     42(5):A2837-A2864, 2020,
%
%   transcribed from the MATLAB listing given there, with one algebraic
%   shortcut.  The listing forms Ub = W'*U' and uses only sum(Ub.^2,1) and
%   Ub(end,:).  Since W is orthogonal it preserves column norms, so
%
%       sum(Ub.^2,1) = sum(U'.^2,1) = the leverage scores of U,
%
%   independent of W and computable once outside the loop, and Ub(end,:) is the
%   single matrix-vector product (U*W(:,end))'.  That replaces an n^2*N product
%   per iteration with an n*N one, which is what makes the n = 1000 case
%   tractable; GPODE_REFERENCE holds the literal transcription for checking.

n   = size(U,2);
ell = sum(U.^2, 2)';                        % leverage scores, = sum(Ub.^2,1)

[~, ~, p] = qr(U', 'vector');
p = p(1:n)';

for i = n+1:m
    [~, S, W] = svd(U(p,:), 0);
    g  = S(end-1,end-1)^2 - S(end,end)^2;
    ub = (U*W(:,end))';                     % = Ub(end,:)

    r = g + ell;
    r = r - sqrt(max((g + ell).^2 - 4*g*ub.^2, 0));

    [~, I] = sort(r, 'descend');
    e = 1;
    while any(I(e) == p)
        e = e + 1;
    end
    p(end+1) = I(e);                        %#ok<AGROW> grows m-n times
end

end
