import pandas as pd

def merge_viability(cell_file="cell_viability.csv", tails_file="3_tails.csv", output_file="merged_output.csv"):
    """
    Merge viability values from cell_viability.csv into rows from 3_tails.csv
    where 'identifier' contains 'a3'. The match is based on 'number' in 3_tails.csv
    and 'CAD LNP No.' in cell_viability.csv.
    """
    # Load both CSVs
    cell_df = pd.read_csv(cell_file)
    tails_df = pd.read_csv(tails_file)

    # Filter rows in tails_df where identifier contains 'a3'
    filtered_tails = tails_df[tails_df['identifier'].str.contains("a3", na=False)]

    # Merge on number (from tails) and CAD LNP No. (from cell_df)
    merged_df = pd.merge(
        filtered_tails,
        cell_df[['CAD LNP No.', 'viability']],
        left_on='number',
        right_on='CAD LNP No.',
        how='left'
    )

    # Drop duplicate CAD LNP No. column (since number already exists)
    merged_df = merged_df.drop(columns=['CAD LNP No.'])

    # Save to new CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved to {output_file}")

# Example usage:
merge_viability("cell_viability.csv", "3_tails.csv", "merged_output.csv")
