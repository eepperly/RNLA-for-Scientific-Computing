function S = sample_no_replace(w, s)
%SAMPLE_NO_REPLACE  Draw s distinct indices with probability proportional to w.
%   S = SAMPLE_NO_REPLACE(w,s) returns s distinct indices into w, drawn by
%   successive sampling proportional to the weights.
%
%   MATLAB's randsample refuses weighted sampling without replacement, and
%   sampling with replacement is not an option here: a repeated index makes
%   U(S,:) singular and DEIM undefined.  This uses the exponential race of
%   Efraimidis and Spirakis: give index i the key -log(U_i)/w_i with U_i
%   uniform, and keep the s smallest keys.  Entries with zero weight get an
%   infinite key and are never drawn.
%
%   See also SELECTLEVERAGE, SELECTLEVERAGEDVOLUME.

w = max(w(:), 0);
n = numel(w);

if s > n
    error('sample_no_replace:TooMany', 's = %d exceeds %d entries.', s, n);
end
if s > nnz(w)
    error('sample_no_replace:TooFewPositive', ...
          'Asked for %d draws but only %d weights are positive.', s, nnz(w));
end

key      = -log(rand(n,1)) ./ w;        % Inf wherever w is zero
[~, ord] = sort(key, 'ascend');
S        = ord(1:s);

end
