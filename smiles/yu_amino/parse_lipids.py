import pandas as pd
import re

main = pd.read_csv("main_data__.csv")
ids = main["Lipid_name"]
main.drop("Lipid_name", axis=1, inplace=True)
main.drop("identifier", axis=1, inplace=True)
main.drop("Unnamed: 0", axis=1, inplace=True)
main.to_csv("main_data.csv", index=False)

# Mapping from prefix to number of tails
TAIL_MAP = {
    "AM": 9,
    "GT": 9,
    "GN": 7,
    "HG": 5
}

def parse_lipid(identifier: str) -> dict:
    """
    Parse a lipid identifier (e.g. 'AM-EP6', 'GN-EP10', 'HG-Ac8-1')
    and return metadata: number of tails and carbons.
    """
    # First two characters determine tails
    prefix = identifier[:2]
    tails = TAIL_MAP.get(prefix, None)

    # Extract carbons: look for digits after '-' (handles EP6, EP10-1, Ac8-1, etc.)
    match = re.search(r'-(?:[A-Za-z]+)?(\d+)', identifier)
    carbons = int(match.group(1)) if match else None

    return {
        "identifier": identifier,
        "tails": tails,
        "carbons": carbons
    }


# Parse all lipids into metadata
parsed_data = [parse_lipid(lipid) for lipid in ids]

# Convert to DataFrame
df = pd.DataFrame(parsed_data)

df.to_csv("individual_metadata.csv", index=False)