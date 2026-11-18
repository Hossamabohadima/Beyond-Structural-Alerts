# Beyond Structural Alerts: Toxicity and Mutagenicity XAI Benchmark

A benchmarking workflow for analyzing the robustness, consistency, and agreement of explainability techniques applied to molecular property prediction models for toxicity and mutagenicity detection.

**Version:** 1.0.0 | **Run Date:** November 18, 2026

## Getting Started

This section will help you set up the required environment and run the project locally.

### Prerequisites

You need to have an environment with Python 3.13. All required dependencies and exact library versions are listed in `requirements.txt` in the root directory.

    $ pip install -r requirements.txt

### Installation

Clone the repository:

    $ git clone https://github.com/Hossamabohadima/Beyond-Structural-Alerts.git
    $ cd Beyond-Structural-Alerts

## Usage

All executable scripts are located inside the `src` directory. Navigate to `src` and run the main package module:

    $ cd src
    $ python -m toxicity

You can optionally specify the hyperparameter optimization budget (in seconds) via CLI:

    $ python -m toxicity --hyperparameter_optimization_time 900

This executes the full pipeline: training the `transformer_matrix` model with Optuna hyperparameter optimization, evaluating on the test set, computing raw XAI attributions, and automatically running the XAI visualization and table export module (`xai_visualization.py`).

### Execution Parameters (configured in `src/toxicity/__main__.py`)

- **task**: The task to execute (`train`).
- **dataset**: Specifies the dataset source. Choices are:
  - `ames`
  - `file` (default in current run; uses custom dataset files specified in `dataset_path`)
- **dataset_path**: Path(s) to custom dataset files (`toxicity/data/train_clean.csv`, `toxicity/data/val_clean.csv`, `toxicity/data/test_clean.csv`). Datasets must contain a `smiles` column and a binary label column.
- **representations**: Molecular representation used (`transformer_matrix`).
- **split**: Split ratios for training, validation, and testing if using a single unsplit file. Default: `[0.7, 0.1, 0.2]`.
- **hyperparameter_optimization_time**: Time in seconds allocated for Optuna hyperparameter optimization. Default: `900` (15 minutes).

## Explainable AI (XAI) Methods

Five post-hoc attribution methods are computed on the trained model to quantify token/atom-level contributions toward the positive class (Class 1: Toxic / Mutagenic):

- **SHAP (`shap.npy`)**: Approximates Shapley values from cooperative game theory to assign each input feature an importance value representing its marginal contribution to the model's prediction.
- **Integrated Gradients (`ig.npy`)**: Computes the path integral of gradients from a baseline input to the actual molecular input, satisfying sensitivity and implementation invariance axioms.
- **DeepLIFT (`deeplift.npy`)**: Compares the activation of each neuron to its reference activation and assigns contribution scores according to the difference, avoiding gradient saturation issues.
- **Occlusion (`occlusion.npy`)**: A perturbation-based method that systematically masks/occludes regions of the input representation and measures the resulting drop or change in the model's output score.
- **Grad-CAM (`gradcams/gradcam_filter_1.npy`)**: Uses the gradients of the target class flowing into convolutional/encoder feature maps to produce a coarse localization map highlighting important molecular regions.

**Attribution Processing Details:**
- **Target**: Attributions are computed with respect to pre-sigmoid **Logits** for **Class 1 (Toxic / Mutagenic)**.
- **Embedding Aggregation**: For 2D token-embedding attribution matrices, feature dimensions are aggregated to a single scalar score per token using the mean across the embedding axis (`sample.mean(axis=-1)`), then mapped sequentially to RDKit molecular atoms for visualization.

## Pairwise Similarity (`Sim`) Definition

In the comparison visualizations (`compare/`) and `pairs.csv`, **`Sim` does NOT represent whole-molecule chemical similarity** (such as Morgan/ECFP fingerprints). Instead:

- **Fingerprint Definition**: Binary structural alert presence vector (`alert_fp`), where each bit indicates the presence (`1`) or absence (`0`) of a specific SMARTS alert from `core_alert_definitions.csv`.
- **Similarity Metric**: **Jaccard / Tanimoto similarity** computed on the binary alert vectors:
  `Sim = |Alert_FP_1 ∩ Alert_FP_2| / |Alert_FP_1 ∪ Alert_FP_2|`
- **Interpretation**: `Sim` strictly measures **structural alert overlap** between two molecules (e.g., `100%` means both molecules trigger the exact same set of structural alerts despite having different toxicity outcomes or model predictions).

## Test Output & Generated Tables

When you run the pipeline from `src/`, the results are saved in the working directory with the following structure:

### `model/`
- `roc.csv`: Raw evaluation file containing true labels (`y_true`) and raw predicted probabilities (`y_pred`) for the test set.
- `training/version_0/hparams.yaml`: Saved model hyperparameters and training configurations.
- `*.ckpt`: Checkpoint file(s) for the trained model.
- `deeplift.npy`, `ig.npy`, `occlusion.npy`, `shap.npy`: Raw numerical attribution arrays (before color normalization) for each XAI method.
- `gradcams/`: Layer-wise Grad-CAM raw `.npy` maps.

### `model/images_xai/` (Generated Tables & Visualizations)

1. **`predictions_test.csv`** *(Full Test Set Predictions — 1,585 compounds)*:
   - Contains predictions for every compound in the test split (not limited to the 40 visualized cases).
   - **Columns**: `compound_id`, `identity_group_id`, `original_smiles`, `standardized_smiles`, `split`, `y_true`, `p_positive` (raw pre-calibration probability), `y_pred` (binary prediction at `threshold = 0.5`), `threshold`, `run_id`, `seed` (plus `source` / `original_record_id` if present in the input file).

2. **`alert_matches.csv`** *(Structural Alert Matching & Atom Indices)*:
   - Records all matched SMARTS structural alerts for every compound in the test set.
   - **Columns**: `orig_idx`, `smiles`, `y_true`, `y_pred`, `num_matched_alerts`, `matched_alerts_smarts`, and `matched_atom_indices_json` (a JSON dictionary mapping each matched SMARTS pattern to a list of RDKit 0-indexed atom index lists, e.g., `{"[C]=[C]-[CX3](=[OX1])": [[3, 2, 1, 0]]}`).

3. **`pairs.csv`** *(Selected Comparison Pairs)*:
   - Documents the highest-similarity compound pairs selected for side-by-side XAI comparison across three contrast cases: `TP_vs_TN`, `TP_vs_FN`, and `FP_vs_TN`.
   - **Columns**: `case`, `mol1_orig_idx`, `mol1_smiles`, `mol1_y_true`, `mol1_y_pred`, `mol2_orig_idx`, `mol2_smiles`, `mol2_y_true`, `mol2_y_pred`, `shared_alerts_smarts`, `similarity`, `fingerprint_definition`, and `similarity_metric`.

4. **`image_mapping.csv`** *(Compound-to-Image Index)*:
   - Maps every generated XAI image back to its corresponding compound(s) and XAI method.
   - **Columns**: `image_type` (`individual` or `comparison`), `case`, `orig_idx`, `smiles`, `y_true`, `y_pred`, `xai_method`, `image_filename`, and `relative_path`.

5. **Image Folders**:
   - `TP/`, `TN/`, `FP/`, `FN/`: Individual molecular XAI heatmaps for the top 10 compounds per confusion-matrix quadrant across all 5 XAI methods.
   - `compare/`: Side-by-side comparison heatmaps for the selected pairs in `pairs.csv`.

### `data_dir/` & `model_config.yaml`
- `data_dir/`: Processed dataset files used during the run.
- `model_config.yaml`: Summary of execution parameters used for the run.

## Run Notes & Deliverables Status

- **Class 1 Definition**: Represents a **Toxic / Mutagenic** compound.
- **Threshold & Calibration**: Standard classification threshold of `0.5`. Output probabilities (`p_positive`) are raw pre-calibration Sigmoid outputs.
- **Random Seed**: No global random seed was strictly enforced for this run (`seed` is left empty in `predictions_test.csv`).
- **Structural Alerts**: Defined in `src/toxicity/data/core_alert_definitions.csv`.
- **Unavailable Artifacts**: Predictions for `train` and `val` splits (only `test` predictions are exported in this run) and a standalone `token-to-atom` mapping export file are not generated in this version.

## Citation

This repository is part of the project "Beyond Structural Alerts: Toxicity and Mutagenicity XAI Benchmark".  
Published paper: TODO  
Bibtex: TODO
