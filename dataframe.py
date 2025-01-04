import pandas as pd
import numpy as np


df = pd.read_csv("breast_cancer.csv")

df.info()

# Convert the column to numeric, forcing non-numeric values to NaN
df["Bare Nuclei"] = pd.to_numeric(df["Bare Nuclei"], errors="coerce")

# Calculate the mean in Class, ignoring NaN values
class_mean_value = df[["Class", "Bare Nuclei"]].groupby("Class").mean()

# Replace NaN values (originally "?") with the Class mean value
df["Bare Nuclei"] = df.apply(lambda row: class_mean_value.loc[row["Class"]].iloc[0] if pd.isna(row["Bare Nuclei"]) else row["Bare Nuclei"], axis=1)

# Convert values to numeric and round, then convert to integer
df["Bare Nuclei"] = df["Bare Nuclei"].round().astype(float).apply(lambda x: int(x) if not pd.isna(x) else np.nan)


# Remove duplicates (all rows with same data)
df.drop_duplicates(inplace=True)
df.info()


# Remove duplicates (all rows with same data)
unique_check = df["Sample code number"].nunique()
# print(unique_check)
df = df.groupby("Sample code number").agg("max").reset_index()


# Save the DataFrame to a new CSV file in overwrite mode
df.to_csv("breast_cancer_upd.csv", index=False, mode="w")


# Dataframe information / Descriptive statistics
df.info()
describe_df = df.describe()
print(describe_df)

# Print the updated DataFrame
print(df)