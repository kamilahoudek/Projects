import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("breast_cancer_upd.csv", usecols=["Class", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"])

# Display the first few rows of the dataframe
# print(df.head())

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

# Display the first five values of X and y
# print(X.values[:5])
# print(y[:5])

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display the model's intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate the R-squared value and MSE value
r2 = r2_score(y_train, y_train_pred)
print("R-squared:", r2)
mse = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error:", mse)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Calculate the R-squared value and MSE value for the test set
r2_test = r2_score(y_test, y_test_pred)
print("R-squared (test set):", r2_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error (test set):", mse_test)

# Create prediction data with the same feature names
predict_data = pd.DataFrame([[5, 2, 2, 2, 10, 10, 3, 2, 2]], columns=X.columns)
print("Predict data:", predict_data)

# Make a prediction
prediction = model.predict(predict_data)
print("Prediction:", prediction)

# Decode the prediction back to the original labels
decoded_prediction = label_encoder.inverse_transform(prediction.round().astype(int))
print("Decoded Prediction:", decoded_prediction)
