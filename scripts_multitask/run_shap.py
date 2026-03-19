import sys
import os
import glob
import pickle
import shutil
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator, DataStructs

# --- Configuration ---
FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 2048 

# --- Helper Functions ---

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def ensure_clean_dir(directory):
    """
    Completely removes the directory if it exists, then recreates it.
    This prevents 'ghost' images from previous runs from appearing.
    """
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
        except Exception as e:
            print(f"Warning: Could not clean directory {directory}: {e}")
    os.makedirs(directory)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_fingerprints_and_info(smiles_list):
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_NBITS)
    
    mols = []
    fps = []
    bit_infos = []
    valid_mols = []

    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp = mfgen.GetFingerprint(m, additionalOutput=ao)
            
            np_fp = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, np_fp)
            
            fps.append(np_fp)
            bit_infos.append(ao.GetBitInfoMap())
            valid_mols.append(m)
            
    return np.array(fps), bit_infos, valid_mols

# --- Visualization Function 1: Contextual ---
def save_context_images(top_bits, X_fp, mols, bit_infos, output_dir):
    # 1. WIPE OLD DATA
    ensure_clean_dir(output_dir)
    
    saved_count = 0
    
    for rank, bit_idx in enumerate(top_bits):
        if saved_count >= 20: break
        
        # 2. SKIP NON-BITS (e.g., Lipid_ID)
        # If the index is larger than the number of bits, it's an extra feature.
        if bit_idx >= FINGERPRINT_NBITS: 
            print(f"  Skipping Rank {rank+1} (Index {bit_idx}): Not a structural bit.")
            continue 

        # Find examples
        examples = []
        for i, fp_row in enumerate(X_fp):
            if fp_row[bit_idx] == 1:
                examples.append((mols[i], bit_infos[i]))
                if len(examples) >= 3: break
        
        if not examples: continue

        try:
            mol, info = examples[0]
            if bit_idx in info:
                atom_idx, radius = info[bit_idx][0]
                
                if radius == 0: continue

                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                highlight_atoms = list(amap.keys())
                
                img_path = os.path.join(output_dir, f"rank_{rank+1}_bit_{bit_idx}.png")
                
                # Draw (B&W Base + Red Highlight)
                dopts = Draw.MolDrawOptions()
                dopts.addAtomIndices = False
                dopts.useBWAtomPalette()
                dopts.highlightBondWidthMultiplier = 8
                
                img = Draw.MolToImage(
                    mol, 
                    size=(400, 300),
                    highlightAtoms=highlight_atoms,
                    highlightColor=(1, 0.4, 0.4), 
                    options=dopts
                )
                img.save(img_path)
                saved_count += 1
        except Exception as e:
            print(f"  Failed context draw bit {bit_idx}: {e}")

# --- Visualization Function 2: Isolated ---
def save_isolated_images(top_bits, X_fp, mols, bit_infos, output_dir):
    # 1. WIPE OLD DATA
    ensure_clean_dir(output_dir)
    
    saved_count = 0
    
    for rank, bit_idx in enumerate(top_bits):
        if saved_count >= 20: break
        
        # 2. SKIP NON-BITS
        if bit_idx >= FINGERPRINT_NBITS: 
            continue 

        examples = []
        for i, fp_row in enumerate(X_fp):
            if fp_row[bit_idx] == 1:
                examples.append((mols[i], bit_infos[i]))
                if len(examples) >= 3: break
        
        if not examples: continue

        try:
            mol, info = examples[0]
            if bit_idx in info:
                atom_idx, radius = info[bit_idx][0]
                if radius == 0: continue

                img_path = os.path.join(output_dir, f"rank_{rank+1}_bit_{bit_idx}.png")
                
                # Draw Standard
                img = Draw.DrawMorganBit(mol, bit_idx, info, useSVG=False)
                
                if isinstance(img, bytes):
                    with open(img_path, "wb") as f: f.write(img)
                elif hasattr(img, 'save'):
                    img.save(img_path)
                else:
                    with open(img_path, "wb") as f: f.write(img.data)
                
                saved_count += 1
        except Exception as e:
            print(f"  Failed isolated draw bit {bit_idx}: {e}")

# --- Core Logic ---

def run_shap_workflow(folder_name):
    base_data_dir = f"../data/crossval_splits/{folder_name}"
    base_results_dir = f"../results/crossval_splits/{folder_name}"
    output_base_dir = f"../results/SHAP/{folder_name}"
    
    print(f"Starting Clean SHAP analysis for: {folder_name}")
    ensure_dir(output_base_dir)

    for cv_idx in range(5): 
        print(f"\n--- Processing Fold {cv_idx} ---")
        fold_output_dir = os.path.join(output_base_dir, f"cv_{cv_idx}")
        ensure_dir(fold_output_dir)

        # Load Data
        model_path = os.path.join(base_data_dir, f"cv_{cv_idx}", f"model_{cv_idx}", "basic_model.pkl")
        if not os.path.exists(model_path): continue
        model = load_pickle(model_path)

        scaler_path = os.path.join(base_data_dir, f"cv_{cv_idx}", f"model_{cv_idx}", "extra_features_scaler.pkl")
        scaler = load_pickle(scaler_path)
        
        pred_csv_path = os.path.join(base_results_dir, "test", f"cv_{cv_idx}", "predicted_vs_actual.csv")
        df_preds = pd.read_csv(pred_csv_path)
        smiles = df_preds['smiles'].tolist()

        extra_feats_path = os.path.join(base_data_dir, "test", "test_extra_x.csv") 
        if not os.path.exists(extra_feats_path):
             extra_feats_path = os.path.join(base_results_dir, "test", "test_extra_x.csv")
        
        df_extra = pd.read_csv(extra_feats_path)
        X_extra = df_extra.values
        X_extra_scaled = scaler.transform(X_extra)

        print("Generating molecular fingerprints...")
        X_fp, bit_infos, valid_mols = generate_fingerprints_and_info(smiles)
        
        X_test = np.hstack([X_fp, X_extra_scaled])
        feat_names = [f"Bit_{i}" for i in range(FINGERPRINT_NBITS)] + list(df_extra.columns)

        print("Calculating SHAP values...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        except:
            background = shap.kmeans(X_test, 10) 
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values_target = shap_values[1] 
        else:
            shap_values_target = shap_values

        # Save Summary Plot
        plt.figure()
        shap.summary_plot(shap_values_target, X_test, feature_names=feat_names, show=False, max_display=20)
        plt.savefig(os.path.join(fold_output_dir, "shap_summary_plot.png"), bbox_inches='tight')
        plt.close()

        # Extract Top 20 Features
        mean_abs_shap = np.mean(np.abs(shap_values_target), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
        
        print("Top 20 Important Features (Indices):", top_indices)
        
        # --- GENERATE IMAGES (Will clean directories first) ---
        dir_context = os.path.join(fold_output_dir, "top_bits_context")
        dir_isolated = os.path.join(fold_output_dir, "top_bits_isolated")
        
        print("  Drawing Contextual Images...")
        save_context_images(top_indices, X_fp, valid_mols, bit_infos, dir_context)
        
        print("  Drawing Isolated Substructures...")
        save_isolated_images(top_indices, X_fp, valid_mols, bit_infos, dir_isolated)
        
        print(f"Fold {cv_idx} analysis complete.")

# --- Generate Report ---

def generate_html_report(folder_name):
    print("\nGenerating Report...")
    base_path = f"../results/SHAP/{folder_name}"
    report_path = os.path.join(base_path, "shap_analysis_report.html")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP Analysis: {folder_name}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }}
            h1 {{ text-align: center; color: #2c3e50; }}
            .fold-section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 40px; }}
            .bits-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; padding: 10px; }}
            .bit-card {{ border: 1px solid #eee; padding: 10px; background: #fff; text-align: center; border-radius: 8px; }}
            .bit-card img {{ max-width: 100%; height: auto; }}
            .bit-caption {{ margin-top: 10px; font-weight: 600; color: #555; font-size: 0.95em; }}
            details {{ margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
            summary {{ background-color: #f1f2f6; padding: 15px; cursor: pointer; font-weight: bold; color: #34495e; outline: none; }}
            summary:hover {{ background-color: #e2e6ea; }}
            details[open] summary {{ border-bottom: 1px solid #ddd; }}
            .details-content {{ padding: 20px; background: #fff; }}
        </style>
    </head>
    <body>
        <h1>SHAP Analysis Report: {folder_name}</h1>
    """

    for cv_idx in range(5):
        fold_dir = os.path.join(base_path, f"cv_{cv_idx}")
        if not os.path.exists(fold_dir): continue

        html += f"""<div class="fold-section"><h2>Cross Validation Fold {cv_idx}</h2>"""
        
        # Summary Plot
        summary_plot = f"cv_{cv_idx}/shap_summary_plot.png"
        html += f"""<div style='text-align:center; margin-bottom: 30px;'><img src="{summary_plot}" style='max-width:80%; border:1px solid #eee;'></div>"""
        
        # Contextual Dropdown
        dir_context = os.path.join(fold_dir, "top_bits_context")
        html += """<details open><summary>Contextual Analysis</summary><div class="details-content"><div class='bits-grid'>"""
        
        if os.path.exists(dir_context):
            files = glob.glob(os.path.join(dir_context, "*.png"))
            try: files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
            except: pass
            
            if not files: html += "<p>No structural bits found in Top 20 for this fold.</p>"

            for img_path in files:
                fname = os.path.basename(img_path)
                parts = fname.replace('.png', '').split('_')
                rank_val = parts[1]
                bit_val = parts[3]
                html += f"""<div class="bit-card"><img src="cv_{cv_idx}/top_bits_context/{fname}" loading="lazy"><div class="bit-caption">Rank #{rank_val} (Bit {bit_val})</div></div>"""
        html += """</div></div></details>"""

        # Isolated Dropdown
        dir_isolated = os.path.join(fold_dir, "top_bits_isolated")
        html += """<details><summary>Isolated Substructures</summary><div class="details-content"><div class='bits-grid'>"""
        
        if os.path.exists(dir_isolated):
            files = glob.glob(os.path.join(dir_isolated, "*.png"))
            try: files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
            except: pass
            
            if not files: html += "<p>No structural bits found in Top 20 for this fold.</p>"

            for img_path in files:
                fname = os.path.basename(img_path)
                parts = fname.replace('.png', '').split('_')
                rank_val = parts[1]
                bit_val = parts[3]
                html += f"""<div class="bit-card"><img src="cv_{cv_idx}/top_bits_isolated/{fname}" loading="lazy"><div class="bit-caption">Rank #{rank_val} (Bit {bit_val})</div></div>"""
        html += """</div></div></details></div>"""

    html += "</body></html>"

    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_shap_clean.py <folder_name>")
        sys.exit(1)
    
    run_shap_workflow(sys.argv[1])
    generate_html_report(sys.argv[1])