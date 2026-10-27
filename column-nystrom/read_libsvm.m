function [y, X] = read_libsvm(path)
%READ_LIBSVM  Read a data file in LIBSVM sparse format.
%   [y,X] = READ_LIBSVM(path) reads lines of the form
%
%       <label> <index1>:<value1> <index2>:<value2> ...
%
%   with 1-based feature indices, and returns the labels as a column vector y
%   and the features as a sparse matrix X with one row per line and
%   max(index) columns.  This matches svm_read_problem(path, return_scipy=True)
%   from the LIBSVM Python package, used by the reference download_data.py:
%   index j becomes column j, and the width is the largest index present in
%   this file, so a training and a test file can come out with different
%   widths.
%
%   See also DOWNLOAD_DATA.

fid = fopen(path, 'r');
if fid < 0
    error('read_libsvm:Open', 'Cannot open %s.', path);
end
C = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
fclose(fid);
L = C{1};
L = L(~cellfun(@isempty, strtrim(L)));      % skip blank lines

n = numel(L);
y = zeros(n, 1);
I = cell(n, 1);  J = cell(n, 1);  V = cell(n, 1);
for i = 1:n
    a    = sscanf(strrep(L{i}, ':', ' '), '%f');
    y(i) = a(1);
    J{i} = a(2:2:end);
    V{i} = a(3:2:end);
    I{i} = repmat(i, numel(J{i}), 1);
end
J = vertcat(J{:});
X = sparse(vertcat(I{:}), J, vertcat(V{:}), n, max([J; 0]));

end
