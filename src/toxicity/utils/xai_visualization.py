
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from PIL import Image

# ==========================================
# 1. Performance Evaluation
# ==========================================
def evaluate_model_performance(roc_file_path):
    print("\n--- Evaluating Model Performance ---")
    try:
        roc_df = pd.read_csv(roc_file_path)
        y_true = roc_df['y_true'].values
        y_pred = roc_df['y_pred'].values
        
        y_pred_binary = (y_pred >= 0.5).astype(float)
        accuracy = np.mean(y_true == y_pred_binary)
        
        desc_score_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]
        
        roc_auc = np.abs(np.trapezoid(tpr, fpr))
        
        print(f"Accuracy:      {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")


# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
def get_alert_info(smiles, alert_smarts, df_alerts):
    if type(smiles) != str: 
        return None, "None"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None, "None"
    
    fp = np.zeros(len(alert_smarts), dtype=int)
    found_alerts = []
    
    for i, smarts in enumerate(alert_smarts):
        if smarts is not None and mol.HasSubstructMatch(smarts):
            fp[i] = 1
            found_alerts.append(df_alerts.iloc[i]['smarts'])
                
    alert_names_str = ", ".join(found_alerts) if found_alerts else "No Alert"
    return fp, alert_names_str

def prepare_data(test_file, roc_file, alerts_file):
    print("\n--- Preparing Data & Alerts ---")
    df_test = pd.read_csv(test_file)
    df_roc = pd.read_csv(roc_file)
    df_alerts = pd.read_csv(alerts_file)

    smiles_col = 'smiles' if 'smiles' in df_test.columns else 'SMILES'
    
    df_test['y_pred'] = df_roc['y_pred'].values
    df_test['y_true'] = df_roc['y_true'].values

    alert_smarts = [Chem.MolFromSmarts(sm) if type(sm) == str else None for sm in df_alerts['smarts']]
    
    alert_results = [get_alert_info(s, alert_smarts, df_alerts) for s in df_test[smiles_col]]
    df_test['alert_fp'] = [res[0] for res in alert_results]
    df_test['alert_names'] = [res[1] for res in alert_results]
    
    return df_test, smiles_col


# ==========================================
# 3. Helper Functions for XAI Images
# ==========================================
def get_atom_weights(smiles, token_scores):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return []
    num_atoms = mol.GetNumAtoms()
    weights = []
    atom_idx = 0
    token_scores = np.atleast_1d(token_scores)
    for i, char in enumerate(smiles):
        if char.isalpha() and atom_idx < num_atoms:
            score = float(token_scores[i]) if i < len(token_scores) else 0.0
            weights.append(score)
            atom_idx += 1
    while len(weights) < num_atoms:
        weights.append(0.0)
    return weights[:num_atoms]

def get_xai_image(mol, weights, size=(400, 400)):
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap='bwr', contourLines=10, draw2d=d)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))

def find_best_pair(df1, df2):
    max_sim = -1
    best_pair = (None, None)
    for i, row1 in df1.iterrows():
        fp1 = row1['alert_fp']
        if fp1 is None: continue
        for j, row2 in df2.iterrows():
            fp2 = row2['alert_fp']
            if fp2 is None: continue
            
            intersection = np.sum(fp1 & fp2)
            union = np.sum(fp1 | fp2)
            sim = intersection / union if union != 0 else 0.0
            
            if sim > max_sim:
                max_sim = sim
                best_pair = (i, j)
    return best_pair, max_sim


# ==========================================
# 4. Generate Individual Case Images
# ==========================================
def generate_individual_cases(df_test, smiles_col, xai_paths, output_dir):
    print("\n--- Generating Individual Case XAI Images ---")
    os.makedirs(output_dir, exist_ok=True)
    
    cases = {
        'TP': df_test[(df_test['y_true'] == 1) & (df_test['y_pred'] >= 0.5)].sort_values(by='y_pred', ascending=False).head(10).reset_index(),
        'TN': df_test[(df_test['y_true'] == 0) & (df_test['y_pred'] < 0.5)].sort_values(by='y_pred', ascending=True).head(10).reset_index(),
        'FP': df_test[(df_test['y_true'] == 0) & (df_test['y_pred'] >= 0.5)].sort_values(by='y_pred', ascending=False).head(10).reset_index(),
        'FN': df_test[(df_test['y_true'] == 1) & (df_test['y_pred'] < 0.5)].sort_values(by='y_pred', ascending=True).head(10).reset_index()
    }

    for case_name, df_case in cases.items():
        print(f"Processing Case: {case_name} (Top {len(df_case)} compounds)")
        case_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        if len(df_case) == 0:
            continue
            
        for i, row in df_case.iterrows():
            orig_idx = row['index']
            smiles = row[smiles_col]
            pred = row['y_pred']
            true_label = row['y_true']
            alerts = row['alert_names']
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            
            is_correct = "Correct Prediction" if (pred >= 0.5 and true_label == 1) or (pred < 0.5 and true_label == 0) else "Incorrect Prediction"
            
            for method_name, path in xai_paths.items():
                if not os.path.exists(path):
                    continue
                    
                try:
                    xai_data = np.load(path)
                    sample = xai_data[orig_idx]
                    
                    scores = sample.mean(axis=-1) if sample.ndim == 2 else sample
                    weights = get_atom_weights(smiles, scores)
                    
                    if len(weights) == mol.GetNumAtoms():
                        fig, ax = plt.subplots(figsize=(6, 6))
                        
                        title = f"Case: {case_name} | {is_correct}\nSMILES: {smiles}\nSMARTS: {alerts}\nPred: {pred:.4f} | True: {true_label}"
                        fig.suptitle(title, fontsize=10, fontweight='bold')
                        
                        img = get_xai_image(mol, weights, size=(400, 400))
                        ax.imshow(img)
                        ax.axis('off')
                        
                        plt.tight_layout()
                        
                        safe_method_name = method_name.replace(' ', '_').replace('-', '_')
                        filename = f"{case_name}_mol_{i+1}_{safe_method_name}.png"
                        save_path = os.path.join(case_dir, filename)
                        
                        plt.savefig(save_path, bbox_inches='tight', dpi=300)
                        plt.close(fig) 
                        
                except Exception as e:
                    print(f"Error processing {method_name} for {smiles}: {e}")


# ==========================================
# 5. Generate Comparison Images
# ==========================================
def generate_comparisons(df_test, smiles_col, xai_paths, output_dir):
    print("\n--- Generating Comparison XAI Images ---")
    comp_output_dir = os.path.join(output_dir, 'compare')
    os.makedirs(comp_output_dir, exist_ok=True)
    
    cases = {
        'TP_vs_TN': {
            'df1': df_test[(df_test['y_true'] == 1) & (df_test['y_pred'] >= 0.5)].reset_index(),
            'df2': df_test[(df_test['y_true'] == 0) & (df_test['y_pred'] < 0.5)].reset_index(),
            'title1': 'Toxic (TP)', 'title2': 'Non-Toxic (TN)'
        },
        'TP_vs_FN': {
            'df1': df_test[(df_test['y_true'] == 1) & (df_test['y_pred'] >= 0.5)].reset_index(),
            'df2': df_test[(df_test['y_true'] == 1) & (df_test['y_pred'] < 0.5)].reset_index(),
            'title1': 'Toxic (TP)', 'title2': 'Toxic (FN - Missed)'
        },
        'FP_vs_TN': {
            'df1': df_test[(df_test['y_true'] == 0) & (df_test['y_pred'] >= 0.5)].reset_index(),
            'df2': df_test[(df_test['y_true'] == 0) & (df_test['y_pred'] < 0.5)].reset_index(),
            'title1': 'Non-Toxic (FP - False Alarm)', 'title2': 'Non-Toxic (TN)'
        }
    }

    for case_name, case_data in cases.items():
        print(f"Processing Pair Case: {case_name}")
        df1 = case_data['df1']
        df2 = case_data['df2']
        
        if len(df1) == 0 or len(df2) == 0:
            continue
            
        (idx1, idx2), sim = find_best_pair(df1, df2)
        if idx1 is None or idx2 is None:
            continue
            
        orig_idx1 = df1.loc[idx1, 'index']
        orig_idx2 = df2.loc[idx2, 'index']
        smiles1 = df1.loc[idx1, smiles_col]
        smiles2 = df2.loc[idx2, smiles_col]
        pred1 = df1.loc[idx1, 'y_pred']
        pred2 = df2.loc[idx2, 'y_pred']
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        for method_name, path in xai_paths.items():
            if not os.path.exists(path):
                continue
                
            try:
                xai_data = np.load(path)
                sample1 = xai_data[orig_idx1]
                sample2 = xai_data[orig_idx2]
                
                scores1 = sample1.mean(axis=-1) if sample1.ndim == 2 else sample1
                scores2 = sample2.mean(axis=-1) if sample2.ndim == 2 else sample2
                
                weights1 = get_atom_weights(smiles1, scores1)
                weights2 = get_atom_weights(smiles2, scores2)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle(f"{case_name.replace('_', ' ')} | XAI: {method_name} | Sim: {sim*100:.0f}%", fontsize=16, fontweight='bold')
                
                if mol1 and len(weights1) == mol1.GetNumAtoms():
                    img1 = get_xai_image(mol1, weights1, size=(450, 450))
                    ax1.imshow(img1)
                    ax1.axis('off')
                    ax1.set_title(f"{case_data['title1']}\nPred: {pred1:.4f}", fontsize=12)
                    
                if mol2 and len(weights2) == mol2.GetNumAtoms():
                    img2 = get_xai_image(mol2, weights2, size=(450, 450))
                    ax2.imshow(img2)
                    ax2.axis('off')
                    ax2.set_title(f"{case_data['title2']}\nPred: {pred2:.4f}", fontsize=12)
                    
                plt.tight_layout()
                
                safe_method_name = method_name.replace(' ', '_').replace('-', '_')
                save_path = os.path.join(comp_output_dir, f"{case_name}_{safe_method_name}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
            except Exception as e:
                print(f"Error processing comparison {method_name}: {e}")

# ==========================================
# 6. Main Runner Function
# ==========================================
def run_xai_visualization(test_file_path, roc_file_path, alerts_file_path, xai_paths_dict, output_dir_base):
    """
    Main function to execute the full XAI visualization pipeline.
    You can call this function from anywhere by passing the required paths.
    """
    evaluate_model_performance(roc_file_path)

    try:
        df_test, smiles_col = prepare_data(test_file_path, roc_file_path, alerts_file_path)
    except FileNotFoundError as e:
        print(f"Initialization Failed: {e}")
        return

    generate_individual_cases(df_test, smiles_col, xai_paths_dict, output_dir_base)
    generate_comparisons(df_test, smiles_col, xai_paths_dict, output_dir_base)
    print(f"\nAll tasks completed successfully. Output saved to: {output_dir_base}")
