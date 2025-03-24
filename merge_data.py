import pandas as pd

# Load the metadata CSV and the sequence CSV
df_meta = pd.read_csv(r"C:\Users\Shijie Wang\Desktop\Research\Protein\Data\pdb_data_no_dups.csv")
df_seq = pd.read_csv(r"C:\Users\Shijie Wang\Desktop\Research\Protein\Data\pdb_data_seq.csv")

# Merge the two DataFrames on 'structureId'
df_merged = pd.merge(df_seq, df_meta, on="structureId", how="inner", suffixes=("_seq", "_meta"))
print("Merged DataFrame columns:")
print(df_merged.columns)

# Filter: Select only rows where the macromoleculeType (from sequence file) equals 'Protein'
df_protein = df_merged[df_merged["macromoleculeType_seq"] == "Protein"]
print("Number of protein entries:", df_protein.shape[0])

# Optionally, you may want to check that the 'residueCount' in both files match:
df_protein = df_protein[df_protein["residueCount_seq"] == df_protein["residueCount_meta"]]

# Save the processed data to a new CSV file for later steps (e.g., structure retrieval and annotation)
df_protein.to_csv("processed_protein_data.csv", index=False)
print("Processed data saved to 'processed_protein_data.csv'.")
