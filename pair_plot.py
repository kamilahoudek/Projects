import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Load the CSV file and select specific columns
df = pd.read_csv("breast_cancer_upd.csv", usecols=["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                                                   "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"])

# Ignore warnings
warnings.filterwarnings("ignore")

# Create the pair plot
sns.pairplot(df, hue="Class")

# Show the plot
plt.show()
