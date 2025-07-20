
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path =  r"D:\Cognify\Content\Dataset .csv" 
data = pd.read_csv(file_path)

# Step 1: Preprocess the Dataset
# Fill missing values
data['Cuisines'] = data['Cuisines'].fillna('Unknown')

# Drop irrelevant columns
irrelevant_columns = [
    'Restaurant ID', 'Restaurant Name', 'Address', 'Locality Verbose',
    'Locality', 'Switch to order menu', 'Rating color', 'Rating text'
]
data = data.drop(irrelevant_columns, axis=1)

# Encode categorical features
categorical_features = ['City', 'Cuisines', 'Currency', 'Has Table booking', 
                        'Has Online delivery', 'Is delivering now']
encoder = LabelEncoder()
for col in categorical_features:
    data[col] = encoder.fit_transform(data[col].astype(str))

# Separate features and target
X = data.drop('Aggregate rating', axis=1)  # Features
y = data['Aggregate rating']  # Target

# Check for non-numeric columns (should be none at this point)
assert len(X.select_dtypes(include=['object']).columns) == 0, "Non-numeric columns remain in the features."

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Train Models
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Step 3: Evaluate Models
# Predictions
lr_predictions = lr_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)

# Metrics
print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, lr_predictions))
print("R-squared:", r2_score(y_test, lr_predictions))

print("\nDecision Tree Regressor:")
print("MSE:", mean_squared_error(y_test, dt_predictions))
print("R-squared:", r2_score(y_test, dt_predictions))

# Step 4: Interpret Results
# Feature importance for Decision Tree
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Step 5: Visualization

# 1. Plot Predictions vs Actual Values
plt.figure(figsize=(12, 6))

# Linear Regression predictions vs Actual
plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_predictions, color='blue', alpha=0.6, label='LR Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Linear Regression: Predictions vs Actual')
plt.legend()

# Decision Tree predictions vs Actual
plt.subplot(1, 2, 2)
plt.scatter(y_test, dt_predictions, color='green', alpha=0.6, label='DT Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Decision Tree Regressor: Predictions vs Actual')
plt.legend()

plt.tight_layout()
plt.show()

# 2. Plot Feature Importances (Decision Tree)
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for Decision Tree Regressor")
plt.gca().invert_yaxis()  # Reverse the order of features
plt.show()
