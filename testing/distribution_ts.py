import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_class_distribution_by_split():
    # --- 1. Load Data ---
    # Ensure you are using the correct column name for your classes (e.g., 'toxicity_class')
    class_col = 'toxicity_class' 
    classes = [0, 1, 2, 3]

    df = pd.read_csv("../data/all_data.csv")
    spec_df = pd.read_csv("distribution.csv", comment="#")
    
    # Create a new column to store the tag (Train vs Split)
    df['split_group'] = 'Unassigned'

    # --- 2. Tag Rows based on Spec File ---
    for _, row in spec_df.iterrows():
        dtypes = row['Data_types_for_component'].split(',')
        vals = row['Values'].split(',')
        group_type = row['Train_or_split'].strip() # 'Train' or 'Split'
        
        # Build a mask to find matching rows
        mask = pd.Series([True] * len(df))
        for i, dtype in enumerate(dtypes):
            col_name = dtype.strip()
            val_name = vals[i].strip()
            
            # Basic type conversion helper (in case csv reads int as str)
            if col_name in df.columns:
                if df[col_name].dtype == 'int64' and val_name.isdigit():
                    mask = mask & (df[col_name] == int(val_name))
                else:
                    mask = mask & (df[col_name].astype(str) == val_name)
        
        # Apply tag
        df.loc[mask, 'split_group'] = group_type

    # --- 3. Filter and Count ---
    # Filter down to the rows that were actually assigned
    assigned_df = df[df['split_group'] != 'Unassigned'].copy()
    
    # Get value counts for Train and Split
    # We use reindex([0, 1, 2]) to ensure all classes show up even if count is 0
    train_counts = assigned_df[assigned_df['split_group'].str.lower() == 'train'][class_col].value_counts().reindex(classes, fill_value=0)
    split_counts = assigned_df[assigned_df['split_group'].str.lower() == 'split'][class_col].value_counts().reindex(classes, fill_value=0)

    # Print counts
    print("\n--- Train Class Counts ---")
    print(train_counts)
    print("\n--- Split Class Counts ---")
    print(split_counts)

    # --- 4. Plotting ---
    x = np.arange(len(classes))
    width = 0.35  # Width of the bars

    plt.figure(figsize=(8, 6))
    
    # Plot bars
    plt.bar(x - width/2, train_counts, width, label='Train', color='skyblue', edgecolor='black')
    plt.bar(x + width/2, split_counts, width, label='Split', color='orange', edgecolor='black')

    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Class Distribution (0, 1, 2) by Split Type")
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels on top of bars
    for i, v in enumerate(train_counts):
        plt.text(i - width/2, v + 1, str(int(v)), ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(split_counts):
        plt.text(i + width/2, v + 1, str(int(v)), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    get_class_distribution_by_split()