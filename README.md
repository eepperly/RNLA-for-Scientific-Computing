# Randomized algorithms in numerical linear algebra for scientific computing

Code for the numerical experiments in

> Ethan N. Epperly, Daniel Kressner, and Yuji Nakatsukasa, *Randomized
> algorithms in numerical linear algebra for scientific computing*, to appear
> in *Acta Numerica* 2027.

The repository contains MATLAB implementations of the randomized algorithms
discussed in the paper and the scripts that produce its figures and tables.

## Requirements

- **MATLAB R2025a or later.** The figure-saving helper `utils/ansave.m` uses
  figure themes (`fig.Theme`, `theme`), which were added in R2025a.
- **Signal Processing Toolbox**, for `dct` (used by `sketching/SRFT.m`).
- **Statistics and Machine Learning Toolbox**, for `randsample`, `datasample`,
  `quantile`, and `pdist2`.
- **Optimization Toolbox**, for `lsqlin` (used by
  `sde/cheb_moment_regression.m`).
- **Python 3 with NumPy, SciPy, and Matplotlib**, only for the two scripts in
  `regularization/` that end in `.py`.

## Running the experiments

Every script that runs an experiment ends in `_experiment` (for example,
`trace/trace_experiment.m`). Everything else is a function used by those
scripts. To run an experiment, open MATLAB and run the script by name or by
path, for example

```matlab
run('trace/trace_experiment.m')
```

Each script adds the folders it needs to the MATLAB path itself, so it can be
run from any working directory.

**Output.** Figures are written with `ansave` as both `.png` and `.fig` files
to a folder named `figures` located *next to* this repository (that is,
`../figures` relative to the repository root), which is where the paper's
LaTeX source looks for them. Create that folder before running, or edit the
`ansave(...)` line at the end of a script to save somewhere else.

**Cached results.** Several of the longer experiments save intermediate
results to `data/` and reuse them on the next run if the parameters have not
changed. The folder is created automatically, and deleting it simply forces a
recomputation.

**Cost.** Some experiments are substantial:

- `sketching/sketching_experiment.m` builds a dense test matrix with 10^6
  rows and needs several gigabytes of memory.
- `index-selection/deim_figure2_experiment.m` takes around ten minutes.
- `column-nystrom/nystrom_table_experiment.m` takes around an hour. It saves
  each dataset's result as it finishes, so it can be interrupted and resumed.

The header comment of each script states its parameters; reduce the problem
sizes or the number of trials there for a quicker run.

**Datasets for the RPCholesky table.** `column-nystrom/nystrom_table_experiment.m`
uses the benchmark datasets of Chen, Epperly, Tropp and Webber, "Randomly
pivoted Cholesky" (CPAM 2025). Download and preprocess them first with

```matlab
download_data   % in column-nystrom/
```

which fetches the data from LIBSVM and OpenML into `data/rpcholesky/`.
`column-nystrom/column_nystrom_experiment.m` uses the UCI Adult data set that
ships with MATLAB (`census1994.mat`).

## Layout

| Folder | Topic | Experiments |
|---|---|---|
| `sketching/` | Sketching matrices and leverage score estimation | `sketching_experiment`, `mp_experiment` |
| `lowrank/` | Randomized Nyström approximation | `nystrom_experiment`, `kernel_experiment` |
| `least-squares/` | Sketch-and-precondition least squares | `sketch_precondition_experiment` |
| `orthogonalization/` | CholeskyQR and randomized CholeskyQR | `cholqr_experiment` |
| `gmres/` | GMRES, sGMRES, and randomized Gram–Schmidt GMRES | `gmres_experiment` |
| `trace/` | Trace estimation (Girard–Hutchinson, Hutch++, XTrace, ...) | `trace_experiment` |
| `sde/` | Spectral density estimation (SLQ, KPM, moment regression) | `sde_experiment` |
| `index-selection/` | Row and column selection, DEIM, interpolative decompositions | `deim_experiment`, `deim_figure2_experiment`, `deim_oversampling_experiment`, `deim_selection_noisy_experiment`, `id_experiment` |
| `column-nystrom/` | Column Nyström approximation and RPCholesky | `column_nystrom_experiment`, `nystrom_table_experiment` |
| `CUR/` | CUR decompositions | `cur_experiment` |
| `regularization/` | Randomized regularization | `diagpert_experiment`, `nonsymm_diagpert_experiment`, `singexample_experiment`, `projection_condition_experiment.py`, `shattering_experiment.py` |
| `utils/` | Plotting helpers (`ansave`, `plot_shaded`) | |

## License

MIT; see [LICENSE](LICENSE).

## Acknowledgment

This code was written with extensive help from Claude Code, using Anthropic's
Claude Opus and Claude Sonnet models.
