import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# Load the CSV file and select specific columns
df = pd.read_csv("breast_cancer_upd.csv", usecols=["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                                                   "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"])


# 1. SCATTER PLOT WITH REGRESSION LINE
# sns.lmplot(x="Clump Thickness", y="Mitoses", data=df)
# plt.title("Scatter Plot with Regression Line")
# plt.show()




# 2. SWARM PLOT
# df = pd.read_csv("breast_cancer_upd.csv", usecols=["Class", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
#                                                    "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"])

# warnings.filterwarnings("ignore")
# plt.figure(figsize=(10, 6))

# sns.swarmplot(x="Uniformity of Cell Size", y="Uniformity of Cell Shape", data=df, hue="Class")

# plt.legend(bbox_to_anchor=(1, 1), loc=2)
# plt.xticks(rotation=15)
# plt.show()



# 3. HEATMAP

# Calculate the correlation matrix and create the heatmap
corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
