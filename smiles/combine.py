import sys
from rdkit import Chem
from rdkit.Chem import AllChem

def attach_tails_to_core(core_smiles, tail_smiles):
    core_mol = Chem.MolFromSmiles(core_smiles)
    tail_mol = Chem.MolFromSmiles(tail_smiles)

    if not core_mol or not tail_mol:
        raise ValueError("Invalid SMILES strings provided.")

    # --- STEP 1: PREPARE THE TAIL (Exactly as before) ---
    tail_dummy = None
    for atom in tail_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            tail_dummy = atom
            break
            
    if not tail_dummy:
        raise ValueError("Tail molecule must contain a wildcard (*)")

    anchor = tail_dummy.GetNeighbors()[0]
    
    # FREEZE VALENCE: Prevent "Ghost Hydrogen" errors on the alkyne
    anchor.SetNumExplicitHs(anchor.GetTotalNumHs())
    anchor.SetNoImplicit(True)
    anchor.SetIntProp("is_anchor", 1)

    # Remove the dummy atom from the tail
    ed_tail = Chem.EditableMol(tail_mol)
    ed_tail.RemoveAtom(tail_dummy.GetIdx())
    tail_frag = ed_tail.GetMol()

    # Re-order atoms so Anchor is Atom 0
    new_anchor_idx = -1
    for atom in tail_frag.GetAtoms():
        if atom.HasProp("is_anchor"):
            new_anchor_idx = atom.GetIdx()
            break
            
    new_order = [new_anchor_idx] + [i for i in range(tail_frag.GetNumAtoms()) if i != new_anchor_idx]
    tail_frag_ordered = Chem.RenumberAtoms(tail_frag, new_order)

    # --- STEP 2: ITERATIVE REPLACEMENT (The Fix) ---
    
    current_mol = core_mol
    dummy_pattern = Chem.MolFromSmarts('[#0]') # Pattern matching any *

    while True:
        # Check if there are any wildcards left
        if not current_mol.HasSubstructMatch(dummy_pattern):
            break
        
        # ReplaceSubstructs with replaceAll=False returns a tuple of possibilities.
        # We take [0] (the first possibility) to replace exactly one wildcard.
        # This updates the graph safely for the next iteration.
        res = Chem.ReplaceSubstructs(current_mol, dummy_pattern, tail_frag_ordered, replaceAll=False)
        
        if not res:
            break
            
        current_mol = res[0]
        
        # Optional: Sanitize in between steps to ensure graph integrity
        try:
            Chem.SanitizeMol(current_mol)
        except:
            pass

    # --- STEP 3: FINAL CLEANUP ---
    try:
        Chem.SanitizeMol(current_mol)
    except Exception as e:
        print(f"Final Sanitization Warning: {e}")

    return Chem.MolToSmiles(current_mol, isomericSmiles=True)

def attach_tails_to_core_old(core_smiles, tail_smiles):
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
    # OLD (Error): '[*:1&!#0][#0].[*:2&!#0][#0]>>[*:1]-[*:2]'
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
    tail_input = "*c1ccc(CCCCCCCCCC)cc1"
    core_input = "*NC(=N)CCCSCCC(=O)OCCCCCCCCCCCCCC"

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