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

This executes the full pipeline: training the `transformer_matrix` model with hyperparameter optimization, evaluating on the test set, computing XAI attributions, and automatically running the XAI visualization module (`xai_visualization.py`).

### Execution Parameters (configured in `src/toxicity/__main__.py`)

- **task**: The task to execute (`train`).
- **dataset**: Specifies the dataset source. Choices are:
  - `ames`
  - `file` (default in current run; uses custom dataset files specified in `dataset_path`)
- **dataset_path**: Path(s) to custom dataset files (`toxicity/data/train_clean.csv`, `toxicity/data/val_clean.csv`, `toxicity/data/test_clean.csv`). Datasets must contain a `smiles` column and a binary label column.
- **representations**: Molecular representation used (`transformer_matrix`).
- **split**: Split ratios for training, validation, and testing if using a single unsplit file. Default: `[0.7, 0.1, 0.2]`.
- **hyperparameter_optimization_time**: Time in seconds allocated for Optuna hyperparameter optimization. Current run: `900` (15 minutes).

## Test Output

When you run the pipeline from `src/`, the results are saved in the working directory with the following structure:

### `model/`
- `roc.csv`: True labels (`y_true`) and raw predicted probabilities (`y_pred`) for the test set.
- `training/version_0/hparams.yaml`: Saved model hyperparameters and training configurations.
- `*.ckpt`: Checkpoint file(s) for the trained model.
- `deeplift.npy`: Raw attribution scores generated using DeepLIFT.
- `ig.npy`: Raw Integrated Gradients attribution scores.
- `occlusion.npy`: Raw Occlusion sensitivity attributions.
- `shap.npy`: Raw SHAP-based feature attributions.
- `gradcams/`: Layer-wise Grad-CAM raw `.npy` maps from the encoder.
- `images_xai/`: Generated molecular XAI visualizations organized by prediction outcome (`TP/`, `TN/`, `FP/`, `FN/`) and pairwise comparisons (`compare/`).

### `data_dir/`
- Contains the processed dataset files used during the run.

### `model_config.yaml`
- A configuration file summarizing all execution parameters used for the run.

## Run Notes & Deliverables Status

- **Class 1 Definition**: Represents a **Toxic / Mutagenic** compound.
- **Threshold & Calibration**: Standard threshold of `0.5`. Output probabilities are raw pre-calibration Sigmoid outputs.
- **XAI Target**: Attributions are computed on pre-sigmoid **Logits** for Class 1.
- **Random Seed**: No global random seed was strictly enforced for this run.
- **Structural Alerts**: Defined in `src/toxicity/data/core_alert_definitions.csv` and `src/toxicity/utils/smarts_patterns.py`.
- **Unavailable Artifacts**: A unified full-split prediction metadata table (`compound_id`, `identity_group_id`, etc.), an explicit exported `token-to-atom` mapping file, and a standalone compound pairs/matched-atom-indices table were not exported in this run version.

## Citation

This repository is part of the project "Beyond Structural Alerts: Toxicity and Mutagenicity XAI Benchmark".  
Published paper: TODO  
Bibtex: TODO
