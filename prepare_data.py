import pandas as pd

DATA_PATH = 'data/raw/incidents.csv'

df = pd.read_csv(DATA_PATH)

print("Dataset Shape: ", df.shape)
print("Column Names: ", df.columns.tolist())

print("First 5 Tickets:\n", df.head(5))
print("\nMissing values:\n", df.isnull().sum())

print("\nAssignment group distribution:")
print(df["Assignment Group"].value_counts())