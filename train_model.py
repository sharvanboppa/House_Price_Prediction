# Import Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# Read CSV File
df = pd.read_csv("dataset/data.csv")
print("Dataset Loaded Successfully!")
print(df.head())
# Data Cleaning
df = df.drop(["date", "street"], axis=1)
df = df.dropna()
df = df[df["price"] > 0]
# Separate Input and Output
X = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "condition",
        "yr_built",
        "city",
        "statezip"
    ]
]
y = df["price"]
# Convert Categorical Data into Numerical Data
X = pd.get_dummies(X, drop_first=True, dtype=int)
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Apply Linear Regression Algorithm
reg = LinearRegression()
reg.fit(X_train, y_train)
print("Model Training Completed!")
# Find Regression Coefficients and Intercept
print("\nRegression Coefficients:", reg.coef_)
print("Regression Intercept:", reg.intercept_)
# Training Performance
y_train_pred = reg.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print("\nTraining Performance")
print("Training R2 Score:", train_r2)
print("Training RMSE:", train_rmse)
# Testing Performance
y_test_pred = reg.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("\nTesting Performance")
print("Testing R2 Score:", test_r2)
print("Testing RMSE:", test_rmse)
# Own Data Point Prediction
new_data = X.iloc[[0]]
prediction = reg.predict(new_data)
print("\nOwn Data Point Prediction")
print("Predicted House Price: $", round(prediction[0], 2))
# Save Model Using Pickle
model_data = {"model": reg,"columns": X.columns.tolist()}
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model_data, file)


