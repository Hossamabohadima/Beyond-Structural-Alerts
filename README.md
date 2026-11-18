# Beyond Structural Alerts: Toxicity and Mutagenicity XAI Benchmark

This repository contains the complete codebase, models, and explainable AI (XAI) outputs for evaluating chemical toxicity and mutagenicity using Deep Learning and structural alerts.

**Run Date:** November 18, 2026  
**Version:** 1.0.0  

---

## 🛠️ Prerequisites and Installation

This project is built using Python 3.13. To ensure reproducibility, all dependencies, library versions, and packages used during this run are documented in the `requirements.txt` file located in the root directory.

**1. Clone the repository:**
`git clone [https://github.com/Hossamabohadima/Beyond-Structural-Alerts.git](https://github.com/Hossamabohadima/Beyond-Structural-Alerts.git)`
`cd Beyond-Structural-Alerts`

**2. Install dependencies:**
`pip install -r requirements.txt`

**3. Run the pipeline:**
All executable scripts are located in the `src` directory. You must navigate to this directory before running the code.
`cd src`
`python -m toxicity`

---

## 📊 Deliverables & Run Documentation

This section provides detailed documentation of the output files, settings, and explicit notes regarding the availability of specific requested deliverables for this run version.

### 1. Scripts and Execution Settings
* **Scripts Provided:** Data processing, splitting, training, prediction, and XAI calculation scripts (Grad-CAM, SHAP, DeepLIFT, Integrated Gradients, Occlusion) are fully available inside the `src` directory.
* **Execution Command:** `python -m toxicity`
* **Model & Training Config:** All model hyperparameters, training configurations, and architecture setups are preserved in `src/model/training/version_0/hparams.yaml`.
* **Saved Model Weights:** Saved in `.ckpt` format inside the `src/model/` directory.
* **Random Seed:** No explicit global random seed (e.g., `pl.seed_everything()`) was strictly enforced or logged for this specific training run.

### 2. Data and Full Predictions
* **Class 1 Definition:** In this project, `Class 1` explicitly refers to a **Toxic** or **Mutagenic** compound.
* **Threshold & Calibration:** A standard threshold of `0.5` is used. The output probabilities (`p_positive`) are raw probabilities taken directly after the Sigmoid activation function (they are **pre-calibration** / uncalibrated).
* **Missing Data Note (Full Prediction Table & Image Mapping):** The unified comprehensive prediction table containing specific columns (`compound_id`, `identity_group_id`, `original_smiles`, `standardized_smiles`, `split`, `y_true`, `p_positive`, `y_pred`, `threshold`, `run_id`, `seed`) and the explicit compound-to-image mapping table are **NOT AVAILABLE** in this run. 
  * *Reason:* The current evaluation pipeline processes the data and outputs the XAI metrics/images directly without aggregating these specific metadata columns into a single unified CSV file. No dummy or synthetic values were generated to fill this gap.

### 3. Raw XAI Attributions
* **Target & Background:** XAI methods were computed using the **Logits** (pre-sigmoid outputs) as the target, interpreting predictions for **Class 1**.
* **Grad-CAM Layer:** Grad-CAM was applied using the final encoder layers of the Transformer model.
* **Format:** Raw, pre-color-normalized numerical attribution values are available and saved as `.npy` arrays in the `src/model/` directory (e.g., `ig.npy`, `shap.npy`, `deeplift.npy`, `gradcams/`).
* **Missing Data Note (Token-to-Atom Map):** The explicit `token-to-atom` mapping matrix is **NOT AVAILABLE** as a saved file in this run.
  * *Reason:* The Transformer utilizes SMILES character/subword tokens rather than explicit graph atoms. The explicit translation mapping was not exported during runtime. However, it can be re-extracted mathematically from the saved tokenizer and model weights without requiring retraining.

### 4. Alerts Definitions and Pairs
* **Alerts Definitions:** The structural alerts utilized in this project are defined in `src/toxicity/data/core_alert_definitions.csv` and `src/toxicity/utils/smarts_patterns.py` (containing SMARTS patterns).
* **Missing Data Note (Pairs Table & Match Indices):** The "Pairs Table" (containing compound pairs, shared alerts, Tanimoto similarity scores, and fingerprints) and the specific "Matched Atom Indices" per compound are **NOT AVAILABLE** in this run package.
  * *Reason:* The pipeline in this specific version executed the deep learning and XAI components but did not execute or export the explicit rule-based similarity matching and pair-generation scripts. No substitute data was generated.
