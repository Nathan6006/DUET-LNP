import sys
from rdkit import Chem
from rdkit.Chem import AllChem

def attach_tails_to_core(core_smiles, tail_smiles):
    """
    Attaches a tail molecule to all wildcard (*) positions on a core molecule.
    
    Args:
        core_smiles (str): SMILES string of the core with wildcards (e.g., "*N(*)C...").
        tail_smiles (str): SMILES string of the tail with one wildcard (e.g., "*C(O)...").
        
    Returns:
        str: SMILES of the final combined molecule.
    """
    
    # 1. Create RDKit Mol objects
    core_mol = Chem.MolFromSmiles(core_smiles)
    tail_mol = Chem.MolFromSmiles(tail_smiles)

    if not core_mol or not tail_mol:
        raise ValueError("Invalid SMILES strings provided.")

    # 2. Define the Reaction using stricter SMARTS
    # NEW (Fix):   '[!#0:1][#0].[!#0:2][#0]>>[*:1]-[*:2]'
    # Explanation:
    # [!#0:1] -> Find any atom that is NOT a dummy (!#0) and map it as 1
    # [#0]    -> It must be bonded to a dummy atom
    rxn_smarts = '[!#0:1][#0].[!#0:2][#0]>>[*:1]-[*:2]'
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)

    current_mol = core_mol
    
    # Sanitize to ensuring standard representations
    Chem.SanitizeMol(current_mol)
    Chem.SanitizeMol(tail_mol)

    # 3. Iteratively attach tails
    max_iterations = 50 
    iteration = 0
    

    while iteration < max_iterations:
        # Check for dummy atoms (Atomic Num 0)
        dummy_atoms = [atom for atom in current_mol.GetAtoms() if atom.GetAtomicNum() == 0]
        num_dummies = len(dummy_atoms)
        

        if num_dummies == 0:
            break

        # Run the reaction: (Current Core) + (Tail)
        products = rxn.RunReactants((current_mol, tail_mol))
        
        if not products:
            print("Error: Reaction failed to find a match, but wildcards remain.")
            # This helps debug if the SMARTS is too restrictive
            print("Debug: Ensure wildcards are explicit dummy atoms (* in SMILES).")
            break
            
        # Update current_mol to be the first product of the first match
        current_mol = products[0][0]
        
        try:
            Chem.SanitizeMol(current_mol)
        except Exception as e:
            print(f"Warning: Sanitization failed at step {iteration}: {e}")
            break
            
        iteration += 1

    if iteration >= max_iterations:
        print("Warning: Reached maximum iterations. Loop stopped to prevent hanging.")

    # 4. Generate final SMILES
    final_smiles = Chem.MolToSmiles(current_mol, isomericSmiles=True)
    return final_smiles

# --- Main Execution ---
if __name__ == "__main__":
    # Example Data
    tail_input = "*N(CCCCCCCC)CCCCCCCC"
    core_input = "**CCOP(=O)(O)OCCCCCCCCC"

    print(f"Core: {core_input}")
    print(f"Tail: {tail_input}")
    print("-" * 30)

    try:
        result_smiles = attach_tails_to_core(core_input, tail_input)
        print("-" * 30)
        print(f"Result: {result_smiles}")
        
        # Verify the result (Formula check)
        result_mol = Chem.MolFromSmiles(result_smiles)
        if result_mol:
            formula = AllChem.CalcMolFormula(result_mol)
            print(f"Final Formula: {formula}")
        
    except Exception as e:
        print(f"An error occurred: {e}")