function F = measure_cdf(grid, nodes, weights)
%MEASURE_CDF  CDF of a discrete measure, evaluated on a sorted grid.
%   F = MEASURE_CDF(grid,nodes,weights) returns F(i), the total weight of the
%   nodes that are <= grid(i).  grid must be sorted ascending.
%
%   Binning rather than interpolation, because the node list is allowed to
%   contain repeats -- the TFIM spectrum is highly degenerate, and quadrature
%   rules can return coincident nodes -- which rules out interp1.

grid    = grid(:);
nodes   = nodes(:);
weights = weights(:);

% Edges [lo, grid(1), ..., grid(end)] so bin j collects (grid(j-1), grid(j)]
% and bin 1 collects everything at or below grid(1).  Nodes above grid(end)
% fall outside and are dropped, which is correct: they are not counted by any
% F(i).
lo = min([nodes; grid]) - 1;
b  = discretize(nodes, [lo; grid]);
ok = ~isnan(b);

acc = accumarray(b(ok), weights(ok), [numel(grid), 1]);
F   = cumsum(acc);

end
