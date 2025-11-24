from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def prepare_fragment(fragment_smiles, anchor_mapnum=777):
    """
    Returns a fragment RDKit Mol with the dummy atom removed and the
    original neighbor (anchor) marked via AtomMapNum=anchor_mapnum.
    """
    frag = Chem.MolFromSmiles(fragment_smiles)
    # Locate the single dummy atom in the fragment
    frag_dummy = [a for a in frag.GetAtoms() if a.GetSymbol() == '*']
    if len(frag_dummy) != 1:
        raise ValueError("Fragment must have exactly one attachment point [*].")
    d = frag_dummy[0]
    # Get the anchor (the atom that was bonded to the dummy)
    nbrs = d.GetNeighbors()
    if len(nbrs) != 1:
        raise ValueError("Fragment dummy atom must have exactly one neighbor.")
    anchor = nbrs[0]
    # Mark the anchor atom so we can find it after removal
    anchor.SetAtomMapNum(anchor_mapnum)

    # Remove the dummy atom (this shifts indices, but we don't care because we marked the anchor)
    frag_edit = Chem.RWMol(frag)
    frag_edit.RemoveAtom(d.GetIdx())

    # Return immutable mol and the anchor index (found by atom map number)
    frag_no_dummy = frag_edit.GetMol()
    # Find the anchor index again by map number
    anchor_idx = None
    for a in frag_no_dummy.GetAtoms():
        if a.GetAtomMapNum() == anchor_mapnum:
            anchor_idx = a.GetIdx()
            break
    if anchor_idx is None:
        raise RuntimeError("Could not locate fragment anchor after dummy removal.")

    return frag_no_dummy, anchor_idx

def combine_smiles_(core_smiles, fragment_smiles):
    """
    For every [*] in core_smiles, attach a copy of fragment_smiles at its [*].
    Returns a combined canonical SMILES.
    """
    core = Chem.MolFromSmiles(core_smiles)
    if core is None:
        raise ValueError("Core SMILES failed to parse.")

    # Prepare the fragment once (without dummy) and record its anchor
    frag_no_dummy, frag_anchor_idx = prepare_fragment(fragment_smiles, anchor_mapnum=777)

    # Work on a mutable copy of the core
    current = Chem.RWMol(core)

    # Collect current dummy atoms (indices will change as we modify; handle in descending order)
    core_dummy_idxs = sorted([a.GetIdx() for a in current.GetAtoms() if a.GetSymbol() == '*'], reverse=True)
    if not core_dummy_idxs:
        raise ValueError("No attachment points [*] found in core.")

    for core_dummy_idx in core_dummy_idxs:
        # Identify the atom to which the core dummy is attached (e.g., N)
        core_dummy_atom = current.GetAtomWithIdx(core_dummy_idx)
        nbrs = core_dummy_atom.GetNeighbors()
        if len(nbrs) != 1:
            raise ValueError("Each core dummy atom must have exactly one neighbor.")
        core_anchor_idx = nbrs[0].GetIdx()

        # Remove the core dummy atom
        current.RemoveAtom(core_dummy_idx)

        # Merge a fresh copy of the fragment (without dummy) into the current core
        # We combine immutable mols, then switch back to RWMol for bonding
        merged = Chem.CombineMols(current.GetMol(), frag_no_dummy)
        merged_edit = Chem.RWMol(merged)

        # The fragment atoms start at this offset in the merged molecule
        offset = current.GetNumAtoms()

        # Connect core anchor to fragment anchor
        merged_edit.AddBond(core_anchor_idx, offset + frag_anchor_idx, Chem.rdchem.BondType.SINGLE)

        # Update current for the next iteration
        current = merged_edit

    # Finalize and sanitize
    combined = current.GetMol()
    Chem.SanitizeMol(combined)
    # Optional: clear atom map numbers used for internal anchoring
    for a in combined.GetAtoms():
        if a.GetAtomMapNum():
            a.SetAtomMapNum(0)

    return Chem.MolToSmiles(combined)

def combine_smiles(core_smiles, fragment_smiles):
    core = Chem.MolFromSmiles(core_smiles)
    if core is None:
        raise ValueError("Core SMILES failed to parse.")

    # Prepare the fragment once, marking its anchor with AtomMapNum=777
    frag_no_dummy, frag_anchor_idx = prepare_fragment(fragment_smiles, anchor_mapnum=777)

    current = Chem.RWMol(core)

    # Find all dummy atoms in the core
    core_dummy_atoms = [a for a in current.GetAtoms() if a.GetSymbol() == '*']
    if not core_dummy_atoms:
        raise ValueError("No attachment points [*] found in core.")

    # Mark each core anchor with a unique AtomMapNum before removing dummy
    for i, dummy in enumerate(core_dummy_atoms):
        nbrs = dummy.GetNeighbors()
        if len(nbrs) != 1:
            raise ValueError("Each core dummy atom must have exactly one neighbor.")
        anchor = nbrs[0]
        anchor.SetAtomMapNum(900 + i)
        current.RemoveAtom(dummy.GetIdx())

    # Attach fragment to each marked anchor
    for i in range(len(core_dummy_atoms)):
        anchor_mapnum = 900 + i

        # Find anchor index by AtomMapNum
        anchor_idx = None
        for a in current.GetAtoms():
            if a.GetAtomMapNum() == anchor_mapnum:
                anchor_idx = a.GetIdx()
                break
        if anchor_idx is None:
            raise RuntimeError(f"Could not find anchor with AtomMapNum {anchor_mapnum}")

        # Merge fragment and bond
        merged = Chem.CombineMols(current.GetMol(), frag_no_dummy)
        merged_edit = Chem.RWMol(merged)

        offset = current.GetNumAtoms()
        merged_edit.AddBond(anchor_idx, offset + frag_anchor_idx, Chem.rdchem.BondType.SINGLE)

        current = merged_edit

    combined = current.GetMol()
    Chem.SanitizeMol(combined)

    # Clear all AtomMapNums
    for a in combined.GetAtoms():
        a.SetAtomMapNum(0)

    return Chem.MolToSmiles(combined)

cores = pd.read_csv("cores.csv")
tails = pd.read_csv("tails.csv")
ep_tails = tails.head(6)
Ac_tails = tails.tail(5)
data = []
for i, core in cores.iterrows(): 
    for h, ep_tail in ep_tails.iterrows():
         identifier = core["Name"] + "-" + ep_tail ["Name"]
         smile = combine_smiles(core["smiles_r"], ep_tail["smiles_r"])
         data.append({"identifier": identifier, "smiles": smile})

for i, core in cores.iterrows(): 
    for h, Ac_tail in Ac_tails.iterrows():
         identifier = core["Name"] + "-" + Ac_tail ["Name"]
         smile = combine_smiles(core["smiles_r"], Ac_tail["smiles_r"])
         data.append({"identifier": identifier, "smiles": smile})

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
           

# Example usage
# tail = "*CC(=O)OCCCCCC"
# core = "*N(*)CC1CCC(N(*)*)C(OC2C(O)C(OC3CCC(C)(O)C(N(*)C)C3O)C(N(*)*)CC2N(*)*)O1"
# full_smiles = combine_smiles(core, tail)
# print("Combined SMILES:", full_smiles)