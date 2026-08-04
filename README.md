# CLRC — Cell-Type-Resolved Ligand-Receptor Connectome

Code accompanying **"Cell-Type-Resolved Ligand-Receptor Connectome of the Human Brain and Its Disruption in Alzheimer's Disease"** (Pak, Hong, et al., *Science Advances*). CLRC reconstructs a directed, cell-type-resolved molecular connectome of the human brain from single-nucleus transcriptomics and curated ligand-receptor databases, tests how well it predicts macroscale structural and functional connectivity, and characterizes its disruption along the Alzheimer's disease continuum.

This repository is a research codebase, not a general-purpose library or CLI tool: it is a linear pipeline of drivers, each taking a YAML config, meant to be read alongside the manuscript's Methods section.

## Summary

Connectomics describes macroscale brain networks; single-cell transcriptomics describes cellular and molecular heterogeneity. Little work has connected the two directly for human brain data. CLRC infers ligand-receptor-mediated communication between (region, cell type) pairs from single-nucleus RNA-seq using a from-scratch Python port of the [NeuronChat](https://github.com/Wei-BioMath/NeuronChat) cell-cell-communication model, builds a connectome-scale feature matrix from those interactions, and:

1. Tests whether inter-regional ligand-receptor signaling predicts structural connectivity (tractography) and functional connectivity (resting-state correlations) via leave-one-brain-region-out (LOBO) gradient-boosted regression.
2. Tests whether individual differences in ligand-receptor signaling relate to Alzheimer's pathology (amyloid, tau) and cognition, using post-mortem ROSMAP data.

## Key capabilities

- **Connectome reconstruction** (`src/neuronchat/`): a from-scratch Python port of NeuronChat's permutation-based cell-cell communication inference, with interchangeable NumPy and PyTorch/CuPy backends for CPU or GPU execution.
- **Feature construction** (`src/clrc/features/`): builds an `(edge, ligand-receptor-pair)` feature matrix from NeuronChat output, aligned to structural/functional connectivity targets and to ROSMAP per-subject data.
- **Structural/functional connectivity prediction** (`src/pipeline/connectivity_prediction/`): Optuna hyperparameter search + LOBO cross-validated XGBoost, SHAP-based interpretation, spatial-autocorrelation null models (brainSMASH-based surrogates), co-expression baselines, and bias-validation cross-prediction.
- **Alzheimer's disease association analysis** (`src/pipeline/pathology_correlation/`): partial Spearman correlation of subject-level ligand-receptor features against amyloid, tau, and cognitive measures, with transcriptomic-covariate adjustment; variance-partition and double-machine-learning explanatory-value analyses.
- **Spatial null models** (`src/clrc/spatial/`): variogram-matched (brainSMASH) surrogate maps for testing whether predictive performance survives control for spatial autocorrelation and proximity.

## Requirements

- **Python 3.12** (pinned via `.python-version`; the project has not been tested on 3.13+).
- **Linux** if using the `gpu` extra (the CUDA-enabled PyTorch/CuPy build pinned in `pyproject.toml` is Linux-only).
- **NeuronChat runs on CPU or GPU**: `src/neuronchat/backends/` provides interchangeable NumPy and PyTorch/CuPy implementations of the same interface. GPU is recommended for the permutation counts (`M`) and region/cell-type scale used in the manuscript, but is not architecturally required.
- **NVIDIA GPU + CUDA 13** if you do want GPU execution — install with the `gpu` extra (`torch`, `cupy-cuda13x`). Edit `[tool.uv.sources]` in `pyproject.toml` and swap `cupy-cuda13x` for `cupy-cuda12x` if targeting CUDA 12.
- **Input data is not bundled.** The ROSMAP (Religious Orders Study / Memory and Aging Project) cohort data is subject to its own restricted-access data use agreements and must be obtained independently (see [Reproducibility](#reproducibility)). `data/` and `out/` are gitignored.

## Quick start

```bash
git clone https://github.com/neuropm-lab/CLRC.git
cd CLRC
uv sync                  # CPU-only: core dependencies
uv sync --extra gpu      # + torch/cupy for GPU-accelerated NeuronChat + XGBoost
uv sync --extra harmonization   # + Cell Type Mapper, for cell-type label harmonization

cp configs/abc_expanded.example.yaml configs/abc_expanded.yaml
cp configs/rosmap_expanded.example.yaml configs/rosmap_expanded.yaml
# edit configs/*.yaml: point data.* paths at your own ABC / ROSMAP downloads
```

Then run a pipeline stage, e.g. building the structural/functional connectivity targets:

```bash
uv run python src/pipeline/shared/build_connectivity_targets.py \
    --config configs/abc_expanded.yaml \
    --target all
```

Every driver accepts `--help` for its exact arguments:

```bash
uv run python src/pipeline/connectivity_prediction/hpo.py --help
```

There is no bundled sample dataset. Every stage operates on ABC/ROSMAP-derived inputs that must be supplied via a config file, per [Requirements](#requirements).

## Installation

Managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                  # core (CPU) dependencies only
uv sync --extra gpu      # add torch + cupy-cuda13x for GPU execution
uv sync --extra harmonization   # add Cell Type Mapper (cell-type label harmonization)
```

`uv sync` respects `.python-version` (pinned to 3.12) and will fetch a matching interpreter automatically if one isn't already installed. `pyproject.toml` + `uv.lock` is the single source of truth for the dependency graph.

The `harmonization` extra is separate because the Allen Institute's [Cell Type Mapper](https://github.com/AllenInstitute/cell_type_mapper) is not published to PyPI and declares the JupyterLab and Sphinx stacks among its runtime dependencies. Keeping it out of the core set means `uv sync` installs 75 packages rather than 250. It is needed only when a dataset's cell-type labels have to be mapped onto a reference taxonomy, which in this study is the ROSMAP source of `build_expression_matrix.py`. Every other stage runs on the core install.

## Usage

All drivers are YAML-config-driven scripts under `src/pipeline/`, organized into three groups. Config paths in the examples below assume you copied and edited the example configs as in [Quick start](#quick-start).

### Shared data preparation (`src/pipeline/shared/`)

```bash
# 1. annotated, cell-type-labelled expression matrix
uv run python src/pipeline/shared/build_expression_matrix.py --config configs/abc_expanded.yaml --source abc

# 2. the connectome itself (NeuronChat permutation inference)
uv run python src/pipeline/shared/build_connectome.py --config configs/abc_expanded.yaml --scope dataset

# 3. structural / functional connectivity targets, and the CCI feature matrix
uv run python src/pipeline/shared/build_connectivity_targets.py --config configs/abc_expanded.yaml --target all
uv run python src/pipeline/shared/build_cci_features.py --config configs/abc_expanded.yaml
```

Steps 1 and 2 are the connectome reconstruction described in *Methods, Construction of a
Cell-Type-Resolved Directed Molecular Connectome*. Both accept a source/scope selector so the
same drivers serve both cohorts: `--source abc|rosmap` and `--scope dataset|subject`. The
ROSMAP arm runs the same two steps with `--source rosmap` and `--scope subject`, producing one
connectome per individual; subject runs are resumable, skipping any subject whose output
already exists.

If you would rather not rebuild the ABC connectome, the reconstructed connectome is deposited at
[doi.org/10.5281/zenodo.21777943](https://doi.org/10.5281/zenodo.21777943) (CC-BY-4.0). Point
`data.nc_h5` at the downloaded file and start from step 3.

### Structural/functional connectivity prediction (`src/pipeline/connectivity_prediction/`)

```bash
# hyperparameter search
uv run python src/pipeline/connectivity_prediction/hpo.py --config configs/abc_expanded.yaml --target sc

# LOBO cross-validated training
uv run python src/pipeline/connectivity_prediction/train_xgboost.py --config configs/abc_expanded.yaml --target sc

# feature importance, SHAP, biological interpretation
uv run python src/pipeline/connectivity_prediction/aggregate_importance.py --config configs/abc_expanded.yaml --target sc
uv run python src/pipeline/connectivity_prediction/shap_analysis.py --config configs/abc_expanded.yaml --target sc
uv run python src/pipeline/connectivity_prediction/cross_target_biology.py --config configs/abc_expanded.yaml
```

Robustness/validation analyses (spatial nulls, co-expression baselines, bias-validation cross-prediction, expression-heterogeneity tests) live alongside these as separate `analyze_*`/`*_baseline`/`spatial_null.py` drivers — run each with `--help` to see its role; each corresponds to a specific robustness check described in the manuscript's Methods.

### Alzheimer's disease association analysis (`src/pipeline/pathology_correlation/`)

```bash
uv run python src/pipeline/pathology_correlation/preprocess_rosmap_lr_subset.py --help
uv run python src/pipeline/pathology_correlation/ad_aggregate.py --config configs/rosmap_expanded.yaml
uv run python src/pipeline/pathology_correlation/ad_correlate.py --config configs/rosmap_expanded.yaml
uv run python src/pipeline/pathology_correlation/region_expression_covariate.py \
    --config configs/rosmap_expanded.yaml --covariate-variant baseline
uv run python src/pipeline/pathology_correlation/explanatory_value_analysis.py --help
```

## Inputs and outputs

**Inputs** (paths configured per-analysis in `configs/*.yaml`):
- ABC atlas snRNA-seq (H5/h5ad), 109 brain regions x 31 cell types.
- DSI Studio tractography `.mat` files and resting-state functional-connectivity `.mat` file (structural/functional connectivity targets).
- ROSMAP snRNA-seq (per-subject H5), clinical/pathology spreadsheet, and subject-linkage CSV.
- `src/neuronchat/data/merged_interactionDB_human*.{csv,json}`: the curated ligand-receptor interaction database, merged from the human subsets of NeuronChatDB and CellChat v2 plus additional curated human entries, and bundled with the repository. It holds 1,092 LR pairs spanning 447 unique HGNC gene symbols. The manuscript's other two counts are computed downstream from this database, not stored: of the 1,092 pairs, 1,014 yield at least one non-zero communication value across the 2,133 populated region-by-cell-type nodes, and 811 retain at least one feature after the NaN/zero pre-selection filter and form the basis of the downstream analyses.

**Outputs**: each driver writes under `<config.output.base_dir>/`, e.g. `out/abc_expanded/` or `out/rosmap_expanded/` (gitignored) — per-fold model artifacts, feature-importance CSVs, SHAP values, spatial-null results, and partial-Spearman correlation tables. Exact output layout is documented in each driver's module docstring.

## Configuration

Two example configs are provided: `configs/abc_expanded.example.yaml` (structural/functional connectivity prediction) and `configs/rosmap_expanded.example.yaml` (Alzheimer's disease association analysis). Copy each to a `.yaml` file without `.example` and edit the `data.*` paths to point at your local copies of the ABC/ROSMAP data. Machine-local absolute paths (e.g. an atlas NIfTI) are expected to be supplied via `${VAR_NAME}` shell-style expansion inside the YAML (see `spatial_null.atlas_nii` in `configs/abc_expanded.example.yaml`).

## Correspondence to the manuscript Methods

Each Methods subsection maps to the files below.

| Methods subsection | Code |
| --- | --- |
| *Dataset 1: Allen Brain Cell Atlas* | `src/pipeline/shared/build_expression_matrix.py --source abc`; `src/clrc/preprocessing/abc.py` |
| *Dataset 2: ROSMAP Multimodal Atlas* | `src/pipeline/shared/build_expression_matrix.py --source rosmap`; `src/clrc/preprocessing/rosmap.py`; `src/pipeline/pathology_correlation/preprocess_rosmap_lr_subset.py` |
| *Construction of a Cell-Type-Resolved Directed Molecular Connectome* | `src/pipeline/shared/build_connectome.py`; `src/clrc/preprocessing/connectome.py`; `src/neuronchat/` (`core.py`, `create.py`, `run.py`, `database.py`, `backends/`) |
| *Allen Human Reference Atlas* | `src/clrc/spatial/atlas.py` |
| *Diffusion-Weighted MRI Structural Connectome* | `src/pipeline/shared/build_connectivity_targets.py --target structural`; `src/clrc/spatial/connectivity_targets.py` |
| *Resting-State Functional MRI Acquisition/Preprocessing* | `src/pipeline/shared/build_connectivity_targets.py --target functional` |
| *Connectivity Prediction Model* | `src/pipeline/connectivity_prediction/hpo.py`, `train_xgboost.py`; `src/clrc/prediction/{hpo,lobo,xgboost,evaluation}.py` |
| *SHAP Analysis* | `src/pipeline/connectivity_prediction/shap_analysis.py`, `aggregate_importance.py`; `src/clrc/prediction/{shap,importance,interpretation}.py` |
| *Differential Expression Analysis* | `src/pipeline/connectivity_prediction/cross_target_biology.py`; `src/clrc/biology/comparison.py` |
| *Gene Set Enrichment Analysis* | `src/clrc/biology/enrichment.py`, `classification.py` |
| *Network Analysis* | `src/clrc/biology/network.py`; `src/pipeline/characterization/` |
| *Additional Computational and Statistical Analyses* | spatial-autocorrelation nulls (`spatial_null.py`, `feature_level_null.py`, `analyze_variogram_null_correlation.py`; `src/clrc/spatial/`), co-expression baselines (`coexpression_baseline.py`, `analyze_coexpression_baseline.py`; `src/clrc/features/coexpression.py`), bias validation by cross-prediction (`cross_prediction.py`, `analyze_cross_prediction.py`; `src/clrc/prediction/bias_validation.py`), expression heterogeneity (`analyze_expression_heterogeneity.py`), bootstrap CIs (`bootstrap_ci.py`; `src/clrc/prediction/bootstrap.py`), variance partition and DML (`src/pipeline/pathology_correlation/explanatory_value_analysis.py`; `src/clrc/causal/explanatory_value.py`) |
| *Alzheimer's disease association* | `src/pipeline/pathology_correlation/ad_aggregate.py`, `ad_correlate.py`, `region_expression_covariate.py`; `src/clrc/ad/` |

## Repository structure

```
src/
  clrc/          Library — connectome feature construction, XGBoost/LOBO prediction,
                 spatial null models, AD partial-Spearman correlation, network/pathway
                 biology, causal (variance-partition, DML) analyses
  neuronchat/    Python port of the NeuronChat cell-cell-communication inference model
                 (numpy + torch/cupy backends); src/neuronchat/data/ holds the curated
                 ligand-receptor interaction database
  pipeline/      YAML-config-driven drivers, split by analysis track:
    shared/               upstream data prep (expression matrix, connectome
                          construction, connectivity targets, CCI feature matrix)
    connectivity_prediction/  SC/FC prediction track (HPO, training, SHAP, robustness checks)
    pathology_correlation/    ROSMAP -> AD association track
configs/         Example YAML configs (copy + edit; production configs are gitignored)
```

## Reproducibility

Note:

- **Installing the software** (`uv sync`) gives you a working environment and every analysis script used to produce the manuscript's results. `.python-version`, `pyproject.toml` and `uv.lock` pin the interpreter and the full resolved dependency graph, so `uv sync` reproduces the exact environment the manuscript was produced in.
- **The tracked example configs carry the manuscript's analysis parameters**, not placeholders: feature-filter thresholds, the XGBoost seed and training settings, the Optuna trial budget, the surrogate count for the spatial nulls, and the clinical variables and covariates for the AD arm are all the values used for the reported results. What you must supply yourself is `data.*` (paths to your own ABC/ROSMAP/connectivity downloads) and `output.base_dir`.
- **Reproducing the reported results** additionally requires: (1) the ABC atlas and ROSMAP snRNA-seq data; (2) running the pipeline stages in the order shown in [Usage](#usage); (3) GPU hardware for the NeuronChat permutation step and XGBoost training at the scale used in the paper (LOBO cross-validation over 101-109 brain regions with Optuna hyperparameter search).

Randomness is seeded, not left to chance: model training uses `xgboost.seed` from the config, and `run_neuronchat` accepts a `seed` that is expanded into a deterministic per-interaction seed so the permutation test reproduces even under parallel execution.

No checkpoints or intermediate result files are bundled in this repository; each is generated by running the corresponding driver.
