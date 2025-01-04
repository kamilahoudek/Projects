import pandas as pd
import numpy as np

df = pd.read_csv("breast_cancer.csv")

# Convert the column to numeric, forcing non-numeric values to NaN
df["Bare Nuclei"] = pd.to_numeric(df["Bare Nuclei"], errors="coerce")

# Calculate the mean/median, ignoring NaN values
mean_value = df["Bare Nuclei"].mean()
print(mean_value)
median_value = df["Bare Nuclei"].median()
print(median_value)


Class_mean = df[["Class", "Bare Nuclei"]].groupby("Class").mean()
print(Class_mean)
Class_median = df[["Class", "Bare Nuclei"]].groupby("Class").median()
print(Class_median)