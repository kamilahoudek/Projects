import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
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

# Apply polynomial features
poly = PolynomialFeatures(degree=2)  # You can adjust the degree based on your needs
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create and train the linear regression model with polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Display the model's intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions on the training set with polynomial features
y_train_pred = model.predict(X_train_poly)

# Calculate the R-squared value and MSE value for the training set
r2_train = r2_score(y_train, y_train_pred)
print("R-squared (training set):", r2_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error (training set):", mse_train)

# Make predictions on the test set with polynomial features
y_test_pred = model.predict(X_test_poly)

# Calculate the R-squared value and MSE value for the test set
r2_test = r2_score(y_test, y_test_pred)
print("R-squared (test set):", r2_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error (test set):", mse_test)

# Create prediction data with polynomial features
predict_data = pd.DataFrame([[3, 2, 2, 2, 4, 4, 4, 1, 1]], columns=X.columns)
predict_data_poly = poly.transform(predict_data)
print("Predict data (polynomial):", predict_data_poly)

# Make a prediction with polynomial features
prediction = model.predict(predict_data_poly)
print("Prediction:", prediction)

# Decode the prediction back to the original labels
decoded_prediction = label_encoder.inverse_transform(prediction.round().astype(int))
print("Decoded Prediction:", decoded_prediction)
