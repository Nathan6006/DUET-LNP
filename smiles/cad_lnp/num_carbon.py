import pandas as pd

# parse the lipid name for num carbon in tail
# Read the input CSV
df = pd.read_csv("individual_metadata.csv")

def parse_lipid_name(name: str) -> int:
    """
    Parse the Lipid_name string:
    - Take the 4th character (index 3) as an integer.
    - If there is a 5th character, subtract 1.
    - If there is a 6th character, subtract an additional 1.
    """
    if not isinstance(name, str) or len(name) < 4:
        return None  # handle malformed entries
    
    try:
        base_val = int(name[5])  # 4th character
            # Adjust based on length
        if len(name) >= 7:
            base_val -= 1
        if len(name) >= 8:
            base_val -= 1
    except ValueError:
        base_val = int(name[6])
        if len(name) >= 8:
            base_val -= 1
        if len(name) >= 9:
            base_val -= 1
    

    
    return base_val

# Apply the parsing function to the Lipid_name column
df["Num_carbon_in_tail"] = df["Lipid_name"].apply(parse_lipid_name)

# Save to new CSV
df.to_csv("individual_metadata_.csv", index=False)