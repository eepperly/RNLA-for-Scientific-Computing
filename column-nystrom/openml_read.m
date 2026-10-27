function [X, y] = openml_read(feat_xml, pq_file, target, remove)
%OPENML_READ  Read an OpenML dataset as openml-python returns it in array format.
%   [X,y] = OPENML_READ(feat_xml,pq_file,target) reads an OpenML
%   dataset from its feature description (features.xml, as served by
%   https://www.openml.org/api/v1/data/features/<id>) and its data file in
%   parquet format, and returns the feature matrix X with the column named
%   target removed and returned separately as y.
%
%   [X,y] = OPENML_READ(...,remove) also removes the named columns in the
%   cell array remove (e.g. an ID column).
%
%   This reproduces what the reference download_data.py receives from
%   openml-python's dataset.get_data(dataset_format="array"):
%
%     * row-identifier and ignore attributes are dropped, all other columns
%       kept in feature-index order;
%     * nominal features become 0-based integer codes, missing as NaN.  The
%       code order is the category order stored in the parquet file when the
%       column is categorical there -- openml-python decodes with pandas, whose
%       .cat.codes follow the file -- and the declared order in the feature
%       description otherwise.  The two usually agree but not always: for
%       hls4ml_lhc_jets_hlf the class column is declared g,q,t,w,z but stored
%       g,q,w,z,t, and the reference output follows the stored order;
%     * the result is rounded to single precision -- openml-python returns
%       float32 in array format -- before being widened back to double, as
%       the reference does.
%
%   See also DOWNLOAD_DATA.

if nargin < 4, remove = {}; end

feat = parse_features(feat_xml);
keep = ~[feat.is_ignore] & ~[feat.is_row_identifier];
feat = feat(keep);
[~, ord] = sort([feat.index]);
feat = feat(ord);

T = parquetread(pq_file, 'VariableNamingRule', 'preserve');
names = {feat.name};
X = zeros(height(T), numel(feat));
for c = 1:numel(feat)
    col = T.(feat(c).name);
    if strcmp(feat(c).data_type, 'nominal')
        X(:,c) = nominal_codes(col, feat(c).nominal);
    else
        X(:,c) = double(col);
    end
end
X = double(single(X));                          % openml-python: float32

tcol = strcmp(names, target);
if ~any(tcol)
    error('openml_read:Target', 'Target column %s not found.', target);
end
y = X(:, tcol);
drop = tcol | ismember(names, remove);
X(:, drop) = [];

end

% -------------------------------------------------------------------------

function codes = nominal_codes(col, values)
%NOMINAL_CODES  0-based category code of each entry.
%   Uses the column's own category order if it is categorical (as pandas
%   would), and the declared values otherwise.
if iscategorical(col)
    values = categories(col);
    s = string(col);
elseif iscellstr(col) || isstring(col) || ischar(col)
    s = string(col);
elseif islogical(col)
    s = string(lower(string(col)));
else
    s = string(col);
end
[tf, loc] = ismember(s, string(values));
codes = loc - 1;
codes(~tf) = NaN;                               % missing or undeclared
end

function feat = parse_features(feat_xml)
%PARSE_FEATURES  Name, index, type and declared nominal values of each feature.
doc  = xmlread(feat_xml);
list = doc.getElementsByTagName('oml:feature');
feat = struct('index', {}, 'name', {}, 'data_type', {}, 'nominal', {}, ...
              'is_ignore', {}, 'is_row_identifier', {});
for i = 0:list.getLength()-1
    f = list.item(i);
    feat(i+1).index = str2double(text_of(f, 'oml:index'));
    feat(i+1).name  = text_of(f, 'oml:name');
    feat(i+1).data_type = text_of(f, 'oml:data_type');
    nv = f.getElementsByTagName('oml:nominal_value');
    vals = cell(1, nv.getLength());
    for j = 0:nv.getLength()-1
        vals{j+1} = char(nv.item(j).getTextContent());
    end
    feat(i+1).nominal = vals;
    feat(i+1).is_ignore = strcmp(text_of(f, 'oml:is_ignore'), 'true');
    feat(i+1).is_row_identifier = strcmp(text_of(f, 'oml:is_row_identifier'), 'true');
end
end

function s = text_of(node, tag)
el = node.getElementsByTagName(tag);
if el.getLength() == 0
    s = '';
else
    s = char(el.item(0).getTextContent());
end
end
