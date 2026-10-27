function download_data(varargin)
%DOWNLOAD_DATA  Download and preprocess the RPCholesky benchmark datasets.
%   DOWNLOAD_DATA() downloads the LIBSVM and OpenML datasets used by Chen,
%   Epperly, Tropp and Webber, "Randomly pivoted Cholesky" (CPAM 2025,
%   arXiv:2207.06503), splits each into training and test sets, and saves
%   <OutDir>/<name>.mat with variables Xtr, Ytr, Xts, Yts.
%
%   This is a MATLAB port of download_data.py from
%   https://github.com/eepperly/Randomly-Pivoted-Cholesky.  As there, a
%   superset of the datasets in the paper's Table 1 is downloaded.
%
%   Options (name-value):
%     'RawDir'       where raw files are downloaded
%                    (default data/rpcholesky/raw)
%     'OutDir'       where the .mat files are written
%                    (default data/rpcholesky/preprocessed)
%     'OpenMLCache'  an openml-python cache directory (containing
%                    <id>/features.xml and <id>/dataset_<id>.pq) to read from
%                    instead of downloading; default ~/.openml/org/openml/www/
%                    datasets if it exists
%     'Only'         cell array of dataset names to process (default: all)
%     'Force'        reprocess datasets whose .mat already exists (default
%                    false)
%
%   A raw LIBSVM file already present in RawDir is not downloaded again.
%
%   OUTPUT FORMAT.  Matches the Python script, so either can feed the
%   experiments: Xtr and Xts are sparse for LIBSVM data and dense double for
%   OpenML data, and Ytr and Yts are row vectors, as scipy's savemat writes
%   1-D arrays.
%
%   DIFFERENCES FROM THE PYTHON SCRIPT.
%     * Random splits.  Datasets with a test_proportion are shuffled and split
%       as sklearn's train_test_split does -- the first ceil(p*n) of a random
%       permutation form the test set, the rest the training set -- with
%       MATLAB's generator seeded at 926.  The permutation is not sklearn's,
%       so for these datasets the train/test assignment differs from the
%       Python output.  The data themselves are identical.
%     * Everything else agrees exactly.  The LIBSVM reader and the OpenML
%       decoder were checked against the Python output for all twenty Table 1
%       datasets: bit-identical where the split is deterministic, and
%       identical as multisets of rows where it is random.
%
%   See also READ_LIBSVM, OPENML_READ, NYSTROM_TABLE_EXPERIMENT.

here = fileparts(mfilename('fullpath'));
p = inputParser;
p.addParameter('RawDir', fullfile(here, '..', 'data', 'rpcholesky', 'raw'));
p.addParameter('OutDir', fullfile(here, '..', 'data', 'rpcholesky', 'preprocessed'));
p.addParameter('OpenMLCache', fullfile(char(java.lang.System.getProperty('user.home')), ...
                                       '.openml', 'org', 'openml', 'www', 'datasets'));
p.addParameter('Only', {});
p.addParameter('Force', false);
p.parse(varargin{:});
opt = p.Results;

if ~isfolder(opt.RawDir), mkdir(opt.RawDir); end
if ~isfolder(opt.OutDir), mkdir(opt.OutDir); end
seed = 926;

% DATASETS.  As in the Python script, a superset of those used in the paper;
% some were dropped from the experiments for having too few points.
% If a dataset has a test URL, the two files are used as given; otherwise it
% is split randomly by test_proportion.
L = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/';
libsvm = { ...
  'a9a',               [L 'binary/a9a'],                          [L 'binary/a9a.t'],                  []; ...
  'cadata',            [L 'regression/cadata'],                   '',                                  0.2; ...
  'cod-rna',           [L 'binary/cod-rna'],                      [L 'binary/cod-rna.t'],              []; ...
  'connect-4',         [L 'multiclass/connect-4'],                '',                                  0.2; ...
  'covtype.binary',    [L 'binary/covtype.libsvm.binary.bz2'],    '',                                  0.2; ...
  'ijcnn1',            [L 'binary/ijcnn1.bz2'],                   [L 'binary/ijcnn1.t.bz2'],           []; ...
  'phishing',          [L 'binary/phishing'],                     '',                                  0.2; ...
  'sensit_vehicle',    [L 'multiclass/vehicle/combined.bz2'],     [L 'multiclass/vehicle/combined.t.bz2'], []; ...
  'sensorless',        [L 'multiclass/Sensorless.scale'],         [L 'multiclass/Sensorless.scale.tr'],[]; ...
  'skin_nonskin',      [L 'binary/skin_nonskin'],                 '',                                  0.2; ...
  'YearPredictionMSD', [L 'regression/YearPredictionMSD.bz2'],    [L 'regression/YearPredictionMSD.t.bz2'], []; ...
  'w8a',               [L 'binary/w8a'],                          [L 'binary/w8a.t'],                  []};

% OpenML: split by test_proportion, or the first ntr points for training and
% the next nts for testing.
openml = { ...
%  name                     id     target        split           remove
  'ACSIncome',              43141, 'PINCP',     {'prop', 0.2},  {}; ...
  'Airlines_DepDelay_1M',   42721, 'DepDelay',  {'prop', 0.2},  {}; ...
  'Click_prediction_small', 1218,  'click',     {'prop', 0.2},  {}; ...
  'COMET_MC_SAMPLE',        23395, 'label',     {'prop', 0.2},  {}; ...
  'creditcard',             1597,  'Class',     {'prop', 0.2},  {}; ...
  'diamonds',               42225, 'price',     {'prop', 0.2},  {}; ...
  'HIGGS',                  42769, 'target',    {'first', 500000, 500000}, {}; ...
  'hls4ml_lhc_jets_hlf',    42468, 'class',     {'prop', 0.2},  {}; ...
  'jannis',                 44131, 'class',     {'prop', 0.2},  {}; ...
  'Medical-Appointment',    43617, 'show',      {'prop', 0.2},  {}; ...
  'MiniBoonNE',             41150, 'signal',    {'prop', 0.2},  {}; ...
  'MNIST',                  554,   'class',     {'first', 60000, 10000}, {}; ...
  'santander',              42435, 'target',    {'prop', 0.2},  {'ID_code'}; ...
  'volkert',                41166, 'class',     {'prop', 0.2},  {}; ...
  'yolanda',                42705, '101',       {'prop', 0.2},  {}};

% --- LIBSVM -------------------------------------------------------------
for d = 1:size(libsvm, 1)
    name = libsvm{d,1};
    if skip(name, opt), continue, end
    fprintf('%s\n', name);

    [Ytr, Xtr] = read_libsvm(fetch_file(libsvm{d,2}, opt.RawDir));
    if ~isempty(libsvm{d,3})
        [Yts, Xts] = read_libsvm(fetch_file(libsvm{d,3}, opt.RawDir));
    else
        [tr, ts] = split_indices(size(Xtr,1), libsvm{d,4}, seed);
        Xts = Xtr(ts,:);  Yts = Ytr(ts);
        Xtr = Xtr(tr,:);  Ytr = Ytr(tr);
    end
    save_split(opt.OutDir, name, Xtr, Ytr, Xts, Yts);
end

% --- OpenML -------------------------------------------------------------
for d = 1:size(openml, 1)
    name = openml{d,1};
    if skip(name, opt), continue, end
    fprintf('%s\n', name);

    [feat_xml, pq] = openml_files(openml{d,2}, opt);
    [X, y] = openml_read(feat_xml, pq, openml{d,3}, openml{d,5});

    sp = openml{d,4};
    if strcmp(sp{1}, 'prop')
        [tr, ts] = split_indices(size(X,1), sp{2}, seed);
    else
        tr = 1:sp{2};
        ts = sp{2} + (1:sp{3});
    end
    save_split(opt.OutDir, name, X(tr,:), y(tr), X(ts,:), y(ts));
end

end

% =========================================================================

function tf = skip(name, opt)
tf = (~isempty(opt.Only) && ~ismember(name, opt.Only)) || ...
     (~opt.Force && isfile(fullfile(opt.OutDir, [name '.mat'])));
end

function [tr, ts] = split_indices(n, test_proportion, seed)
%SPLIT_INDICES  Shuffled split in the manner of sklearn's train_test_split:
%   the first ceil(p*n) entries of a random permutation are the test set.
rng(seed);
perm = randperm(n);
nts  = ceil(test_proportion * n);
ts   = perm(1:nts);
tr   = perm(nts+1:end);
end

function save_split(outdir, name, Xtr, Ytr, Xts, Yts)
Ytr = Ytr(:)';  Yts = Yts(:)';                  % row vectors, as savemat writes
save(fullfile(outdir, [name '.mat']), 'Xtr', 'Ytr', 'Xts', 'Yts', '-v7');
end

function path = fetch_file(url, rawdir)
%FETCH_FILE  Download url into rawdir unless already present, decompressing
%   .bz2 and .xz files.  Returns the path of the (decompressed) file.
[~, base, ext] = fileparts(url);
fname = [base ext];
if any(strcmpi(ext, {'.bz2', '.xz'}))
    plain = fullfile(rawdir, base);
else
    plain = fullfile(rawdir, fname);
end
if isfile(plain)
    path = plain;
    return
end

fprintf('  Downloading %s\n', url);
dest = fullfile(rawdir, fname);
websave(dest, url, weboptions('Timeout', 600));

switch lower(ext)
    case '.bz2', cmd = sprintf('bunzip2 -f "%s"', dest);
    case '.xz',  cmd = sprintf('xz -d -f "%s"', dest);
    otherwise,   cmd = '';
end
if ~isempty(cmd)
    status = system(cmd);
    if status ~= 0
        error('download_data:Decompress', 'Could not decompress %s.', dest);
    end
end
path = plain;
end

function [feat_xml, pq] = openml_files(id, opt)
%OPENML_FILES  Feature description and parquet data for OpenML dataset id,
%   from the openml-python cache if available, otherwise downloaded through
%   the OpenML REST API into RawDir/openml/<id>.
sid   = num2str(id);
cache = fullfile(opt.OpenMLCache, sid);
feat_xml = fullfile(cache, 'features.xml');
pq       = fullfile(cache, ['dataset_' sid '.pq']);
if isfile(feat_xml) && isfile(pq)
    return
end

odir = fullfile(opt.RawDir, 'openml', sid);
if ~isfolder(odir), mkdir(odir); end
feat_xml = fullfile(odir, 'features.xml');
desc_xml = fullfile(odir, 'description.xml');
pq       = fullfile(odir, ['dataset_' sid '.pq']);
wo = weboptions('Timeout', 600);
api = 'https://www.openml.org/api/v1/data/';
if ~isfile(desc_xml), websave(desc_xml, [api sid], wo); end
if ~isfile(feat_xml), websave(feat_xml, [api 'features/' sid], wo); end
if ~isfile(pq)
    doc = xmlread(desc_xml);
    url = char(doc.getElementsByTagName('oml:parquet_url').item(0).getTextContent());
    fprintf('  Downloading %s\n', url);
    websave(pq, url, wo);
end
end
