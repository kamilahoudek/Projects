import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV file
df = pd.read_csv("breast_cancer_upd.csv", usecols=["Class", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"])

# "Class" is the name of the column we want to predict
X = df.drop("Class", axis=1)
y = df["Class"]

# Encode the target column
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print(X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate the R-squared value and MSE value for the training set
r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
print("R-squared (training set):", r2_train)
print("Mean Squared Error (training set):", mse_train)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Calculate the R-squared value and MSE value for the test set
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print("R-squared (test set):", r2_test)
print("Mean Squared Error (test set):", mse_test)

# Create prediction data with the same feature names
predict_data = pd.DataFrame([[3, 2, 2, 2, 3, 3, 3, 6, 2]], columns=X.columns)
print("Predict data:", predict_data)

# Make a prediction
prediction = model.predict(predict_data)
print("Prediction:", prediction)

# Decode the prediction back to the original labels
decoded_prediction = label_encoder.inverse_transform(prediction)
print("Decoded Prediction:", decoded_prediction)
