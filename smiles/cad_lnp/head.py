from rdkit import Chem

def add_head_attachment(df):
    new_smiles = []

    for smi in df["full"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            new_smiles.append(None)
            continue

        rw_mol = Chem.RWMol(mol)
        replaced = False

        for atom in rw_mol.GetAtoms():
            if atom.GetSymbol() == "N":
                print("18")
                if atom.GetDegree() == 1 and atom.GetImplicitValence() == 2:
                    print("20")
                    star = rw_mol.AddAtom(Chem.Atom(0))  # atomic num 0 = '*'
                    rw_mol.AddBond(atom.GetIdx(), star, Chem.BondType.SINGLE)
                    replaced = True
                    break

        if replaced:
            new_smiles.append(Chem.MolToSmiles(rw_mol))
        else:
            print("no NH2 found")
            new_smiles.append(smi)

    df["smiles"] = new_smiles
    return df
