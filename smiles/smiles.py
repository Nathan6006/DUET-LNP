# save as build_lipids.py
import pandas as pd
from rdkit import Chem
from typing import Optional
from head import add_head_attachment


# ----------------------------- helpers -----------------------------


def load_parts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"identifier", "full", "smiles"}
    if not req.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(req))}")

    keep = []
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(str(r["smiles"]))
        if m is None:
            print(f"Skipping invalid SMILES in {path}: {r['identifier']} -> {r['smiles']}")
        else:
            dummies = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
            if len(dummies) != 1:
                print(f"Skipping (needs exactly one [*]) {path}: {r['identifier']} -> {r['smiles']}")
            else:
                nbh = m.GetAtomWithIdx(dummies[0]).GetNeighbors()
                if len(nbh) != 1:
                    print(f"Skipping (dummy must have 1 neighbor) {r['identifier']}")
                else:
                    keep.append(r)
    return pd.DataFrame(keep)



def fuse_on_dummy(head_smiles: str, tail_smiles: str) -> Optional[Chem.Mol]:
    """
    join head and tail by connecting the neighbors of their [*] atoms.
    then delete the [*] atoms.
    """
    mh = Chem.MolFromSmiles(head_smiles)
    mt = Chem.MolFromSmiles(tail_smiles)
    if mh is None or mt is None:
        return None

    dh = next((a.GetIdx() for a in mh.GetAtoms() if a.GetAtomicNum() == 0), None)
    dt = next((a.GetIdx() for a in mt.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dh is None or dt is None:
        return None

    nh = mh.GetAtomWithIdx(dh).GetNeighbors()
    nt = mt.GetAtomWithIdx(dt).GetNeighbors()
    if len(nh) != 1 or len(nt) != 1:
        return None
    nh_idx = nh[0].GetIdx()
    nt_idx = nt[0].GetIdx()

    combo = Chem.CombineMols(mh, mt)
    em = Chem.EditableMol(combo)
    offset = mh.GetNumAtoms()

    em.AddBond(nh_idx, nt_idx + offset, Chem.BondType.SINGLE)
    merged = em.GetMol()

    rw = Chem.RWMol(merged)
    for idx in sorted([dh, dt + offset], reverse=True):
        rw.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(rw)
    except Exception as e:
        print(f"⚠️  Sanitize failed after fusion: {e}")
        return None

    # optional: verify it's one connected component
    frags = Chem.GetMolFrags(rw, asMols=False)
    if len(frags) != 1:
        print("⚠️  Fusion produced multiple fragments (check attachment points).")
        return None

    return rw


def build_all(heads_df: pd.DataFrame, tails_df: pd.DataFrame, out_path: str = "lipids.csv"):
    rows = []
    for _, h in heads_df.iterrows():
        for _, t in tails_df.iterrows():
            ident = f"{h['identifier']}-{t['identifier']}"
            mol = fuse_on_dummy(h["smiles"], t["smiles"])
            if mol is None:
                rows.append({"identifier": ident, "smiles": "ERROR"})
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True)
            rows.append({"identifier": ident, "smiles": smiles})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(rows)} rows.")

# ------------------------------ main -------------------------------

def main():
    df = pd.read_csv("heads.csv", index_col=0)
    new = add_head_attachment(df)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    new.to_csv("heads.csv")
    heads = load_parts("heads.csv")

    tails = load_parts("tails.csv")
    if heads.empty or tails.empty:
        raise SystemExit("No valid heads or tails after validation. Check your [*] markers.")
    build_all(heads, tails, "lipids.csv")

if __name__ == "__main__":
    main()