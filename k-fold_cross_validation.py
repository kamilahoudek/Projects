import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, mean_squared_error

# Load the CSV file
df = pd.read_csv("breast_cancer_upd.csv", usecols=["Class", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"])

# "Class" is the name of the column we want to predict
X = df.drop("Class", axis=1)
y = df["Class"]

# Encode the target column
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define the model
model = DecisionTreeClassifier(random_state=42)

# Define k-fold cross-validation with k=10
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

# Print the cross-validation scores and their mean
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", np.mean(cv_scores))
