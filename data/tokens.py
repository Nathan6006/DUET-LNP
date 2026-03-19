import json
import pandas as pd
import numpy as np
from tqdm import tqdm

CSV_PATH = "all_data.csv"
VOCAB_PATH = "vocab.json"
SMILES_COL = "smiles"

# Load vocab and pre-calculate lengths for efficiency
with open(VOCAB_PATH) as f:
    vocab = json.load(f)

# Sort tokens by length (longest first) to ensure greedy matching
# This is the "secret sauce" for speed
TOKENS_BY_LEN = sorted(vocab.keys(), key=len, reverse=True)
MAX_TOKEN_LEN = len(TOKENS_BY_LEN[0]) if TOKENS_BY_LEN else 0
UNK_TOKEN = "[UNK]"

def tokenize_smiles(smiles):
    tokens = []
    i = 0
    n = len(smiles)
    while i < n:
        match = None
        # Only check up to the maximum possible token length in your vocab
        # This prevents the O(N^2) search on long SMILES strings
        end_search = min(i + MAX_TOKEN_LEN, n)
        
        for j in range(end_search, i, -1):
            piece = smiles[i:j]
            if piece in vocab: # Dict lookup is O(1)
                match = piece
                break
        
        if match:
            tokens.append(match)
            i += len(match)
        else:
            tokens.append(UNK_TOKEN)
            i += 1
    return tokens
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# ... [Keep the previous tokenize_smiles and setup code] ...

def run_sanity_checks(smiles, tokens, unk_token="[UNK]"):
    """Validates that tokens accurately represent the source string."""
    # 1. Reconstruction Check: Do tokens combined equal the original?
    # We remove [UNK] from tokens for this comparison if we want to see 
    # if the known parts match, but usually, we want a 1:1 match.
    reconstructed = "".join([t for t in tokens if t != unk_token])
    
    # If no UNKs, it should be an exact match
    if unk_token not in tokens:
        if reconstructed != smiles:
            return False, "Mismatch: Reconstruction does not match original."
    else:
        # If there are UNKs, the reconstructed string should be a subset
        if len(reconstructed) > len(smiles):
            return False, "Error: Tokens longer than original SMILES."
            
    return True, None

def main():
    df = pd.read_csv(CSV_PATH, usecols=[SMILES_COL])
    smiles_list = df[SMILES_COL].dropna().astype(str).tolist()
    
    lengths = []
    unk_counts = 0
    total_tokens = 0
    failures = 0

    print(f"Tokenizing {len(smiles_list)} SMILES strings...")
    
    # Check a subset for deep validation to save time, or all if needed
    for s in tqdm(smiles_list):
        tokens = tokenize_smiles(s)
        
        # Validation 1: Reconstruction
        is_valid, error_msg = run_sanity_checks(s, tokens, UNK_TOKEN)
        if not is_valid:
            print(f"\n[!] Sanity Check Failed for: {s}")
            print(f"    Tokens: {tokens}")
            failures += 1
        
        # Validation 2: Track [UNK] tokens
        unks_in_s = tokens.count(UNK_TOKEN)
        unk_counts += unks_in_s
        total_tokens += len(tokens)
        lengths.append(len(tokens))

    # --- Final Report ---
    lengths = np.array(lengths)
    unk_rate = (unk_counts / total_tokens) * 100 if total_tokens > 0 else 0

    print("\n" + "="*30)
    print("      SANITY CHECK REPORT")
    print("="*30)
    print(f"Total SMILES Processed: {len(smiles_list)}")
    print(f"Reconstruction Failures: {failures}")
    print(f"Total [UNK] Tokens:     {unk_counts}")
    print(f"Unk Token Rate:         {unk_rate:.2f}%")
    
    if unk_rate > 5.0:
        print(">> WARNING: High [UNK] rate! Check if your vocab is missing common atoms or symbols.")
    
    if failures == 0 and unk_rate < 1.0:
        print(">> STATUS: PASS")
    else:
        print(">> STATUS: REVIEW REQUIRED")
    print("="*30)

    # Print percentiles as before...
    print(f"\nMax token length: {lengths.max()}")
    for p in [75, 80, 90, 95, 99, 99.5]:
        print(f"{p}th percentile: {np.percentile(lengths, p):.1f}")

if __name__ == "__main__":
    main()