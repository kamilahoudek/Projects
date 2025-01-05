import pandas as pd
import numpy as np


df = pd.read_csv("breast_cancer.csv")

# Dataframe information / Descriptive statistics
df.info()
describe_df = df.describe()
print(describe_df)


# Convert the column to numeric, forcing non-numeric values to NaN
df["Bare Nuclei"] = pd.to_numeric(df["Bare Nuclei"], errors="coerce")

# Calculate the mean in Class, ignoring NaN values
class_mean_value = df[["Class", "Bare Nuclei"]].groupby("Class").mean()

# Replace NaN values (originally "?") with the Class mean value
df["Bare Nuclei"] = df.apply(lambda row: class_mean_value.loc[row["Class"]].iloc[0] if pd.isna(row["Bare Nuclei"]) else row["Bare Nuclei"], axis=1)

# Convert values to numeric and round, then convert to integer
df["Bare Nuclei"] = df["Bare Nuclei"].round().astype(float).apply(lambda x: int(x) if not pd.isna(x) else np.nan)


# Remove duplicates (all rows with the same data, included Sample number), reindex DataFrame
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Remove first column with Sample number, as it is no longer needed
df = df.drop(df.columns[0], axis=1)

df.info()

# Save the DataFrame to a new CSV file in overwrite mode
df.to_csv("breast_cancer_upd.csv", index=False, mode="w")

# Print the updated DataFrame
print(df)