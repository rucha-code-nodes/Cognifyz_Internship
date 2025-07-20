
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import classification_report, accuracy_score
# from imblearn.over_sampling import SMOTE

# # Load the dataset
# file_path = r"D:\Cognify\Content\Dataset .csv"
# data = pd.read_csv(file_path)

# # Step 1: Handle missing values
# data['Cuisines'] = data['Cuisines'].fillna('Unknown')

# # Step 2: Drop irrelevant columns
# columns_to_drop = [
#     'Restaurant ID', 'Restaurant Name', 'Address', 'Locality',
#     'Locality Verbose', 'Longitude', 'Latitude', 'Currency',
#     'Rating color', 'Rating text'
# ]
# processed_data = data.drop(columns=columns_to_drop)

# # Step 3: Encode categorical variables with one-hot encoding
# categorical_columns = ['City', 'Has Table booking', 'Has Online delivery',
#                        'Is delivering now', 'Switch to order menu']

# # One-hot encode categorical columns
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# encoded_features = pd.DataFrame(encoder.fit_transform(processed_data[categorical_columns]))
# encoded_features.columns = encoder.get_feature_names_out(categorical_columns)

# # Add one-hot encoded features to the dataset
# processed_data = pd.concat([processed_data.drop(columns=categorical_columns), encoded_features], axis=1)

# # Step 4: Handle target variable (group rare classes)
# min_class_samples = 20
# cuisine_counts = processed_data['Cuisines'].value_counts()
# processed_data['Cuisines'] = processed_data['Cuisines'].apply(
#     lambda x: x if cuisine_counts[x] >= min_class_samples else 'Other'
# )

# # Step 5: Split data into features and target
# X = processed_data.drop(columns=['Cuisines'])
# y = processed_data['Cuisines']

# # Step 6: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# # Step 7: Apply SMOTE for oversampling
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# # Step 8: Train a Random Forest Classifier
# clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15, class_weight='balanced')
# clf.fit(X_resampled, y_resampled)

# # Step 9: Evaluate the model
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# # Generate classification report
# report = classification_report(y_test, y_pred, zero_division=0)

# # Print results
# print(f"Accuracy: {accuracy:.4f}")
# print("\nClassification Report:")
# print(report)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Load the dataset
file_path = r"D:\Cognify\Content\Dataset .csv"
data = pd.read_csv(file_path)

# Step 1: Handle missing values
data['Cuisines'] = data['Cuisines'].fillna('Unknown')

# Step 2: Drop irrelevant columns
columns_to_drop = [
    'Restaurant ID', 'Restaurant Name', 'Address', 'Locality',
    'Locality Verbose', 'Longitude', 'Latitude', 'Currency',
    'Rating color', 'Rating text'
]
processed_data = data.drop(columns=columns_to_drop)

# Step 3: Encode categorical variables with one-hot encoding
categorical_columns = ['City', 'Has Table booking', 'Has Online delivery',
                       'Is delivering now', 'Switch to order menu']

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_features = pd.DataFrame(encoder.fit_transform(processed_data[categorical_columns]))
encoded_features.columns = encoder.get_feature_names_out(categorical_columns)

# Add one-hot encoded features to the dataset
processed_data = pd.concat([processed_data.drop(columns=categorical_columns), encoded_features], axis=1)

# Step 4: Handle target variable (group rare classes)
min_class_samples = 20
cuisine_counts = processed_data['Cuisines'].value_counts()
processed_data['Cuisines'] = processed_data['Cuisines'].apply(
    lambda x: x if cuisine_counts[x] >= min_class_samples else 'Other'
)

# Step 5: Split data into features and target
X = processed_data.drop(columns=['Cuisines'])
y = processed_data['Cuisines']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 7: Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 8: Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15, class_weight='balanced')
clf.fit(X_resampled, y_resampled)

# Step 9: Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Print results to terminal
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 10: Visualize results using Plotly
report_df = pd.DataFrame(report).transpose()
report_df = report_df.iloc[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'

fig = px.bar(
    report_df,
    x=report_df.index,
    y=['precision', 'recall', 'f1-score'],
    barmode='group',
    title="Classification Metrics for Each Class",
    labels={"value": "Score", "index": "Class", "variable": "Metric"}
)

# Show interactive bar graph
fig.show()
