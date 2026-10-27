function p = gpode_reference(U, m)
%GPODE_REFERENCE  Literal transcription of GappyPOD+E, for checking GPODE.
%   Algorithm 1 of Peherstorfer, Drmac and Gugercin (2020), copied from the
%   MATLAB listing in the paper without modification.  Kept only so that the
%   faster GPODE can be checked against it; it is too slow to use in the sweep,
%   forming an n-by-N product every iteration.

[~, ~, p] = qr(U', 'vector');
p = p(1:size(U,2))';
for i = length(p)+1:m
    [~, S, W] = svd(U(p,:), 0);
    g  = S(end-1,end-1).^2 - S(end,end)^2;
    Ub = W'*U';
    r  = g + sum(Ub.^2, 1);
    r  = r - sqrt((g + sum(Ub.^2,1)).^2 - 4*g*Ub(end,:).^2);
    [~, I] = sort(r, 'descend');
    e = 1;
    while any(I(e) == p)
        e = e + 1;
    end
    p(end+1) = I(e);                        %#ok<AGROW>
end
end
