
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MinMaxScaler

# # Load the dataset
# file_path = r"D:\Cognify\Content\Dataset .csv"
# data = pd.read_csv(file_path)

# # Fill missing values in 'Cuisines'
# data['Cuisines'] = data['Cuisines'].fillna('Unknown')

# # Save original columns for display
# original_columns = data[['Cuisines', 'City']]

# # Drop irrelevant or textual columns
# columns_to_drop = [
#     'Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose',
#     'Switch to order menu', 'Rating color', 'Rating text'
# ]
# data = data.drop(columns=columns_to_drop, errors='ignore')

# # Encode categorical features
# categorical_features = ['City', 'Cuisines', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now']
# data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# # Filter out rows with extreme values or low ratings/votes
# data = data[(data['Aggregate rating'] > 2.0) & (data['Votes'] > 10)]
# data = data[(data['Average Cost for two'] > 100) & (data['Average Cost for two'] < 50000)]

# # Normalize numeric features
# scaler = MinMaxScaler()
# data[['Average Cost for two', 'Votes', 'Aggregate rating']] = scaler.fit_transform(
#     data[['Average Cost for two', 'Votes', 'Aggregate rating']]
# )

# # User preferences
# user_preferences = {
#     'Price range': 2,  # Scaled price range
#     'Average Cost for two': 0.5,  # Scaled value between 0 and 1
#     'Aggregate rating': 0.8,  # Scaled value between 0 and 1
#     'Votes': 0.7  # Scaled value between 0 and 1
# }

# # Add dummy entries for all encoded categorical features in user preferences
# for col in data.columns:
#     if col not in user_preferences:
#         user_preferences[col] = 0

# # Convert user preferences to DataFrame
# user_pref_df = pd.DataFrame([user_preferences])

# # Ensure columns match between the dataset and user preferences
# user_pref_df = user_pref_df.reindex(columns=data.columns, fill_value=0)

# # Calculate similarity (exclude 'Similarity' column if present)
# similarity = cosine_similarity(data.drop(columns=['Similarity'], errors='ignore'), user_pref_df)
# data['Similarity'] = similarity

# # Sort and get the top recommendations
# recommendations = data.sort_values(by='Similarity', ascending=False).head(5)

# # Attach original columns for display
# recommendations = recommendations.join(original_columns)

# # Display additional relevant columns for recommendations
# columns_to_display = [
#     'Cuisines', 'City', 'Price range', 'Average Cost for two', 
#     'Aggregate rating', 'Votes', 'Similarity'
# ]
# recommendations_display = recommendations[columns_to_display]

# # Show the top restaurant recommendations
# print("Top Restaurant Recommendations:")
# print(recommendations_display)




import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Load the dataset
file_path = r"D:\Cognify\Content\Dataset .csv"
data = pd.read_csv(file_path)

# Fill missing values in 'Cuisines'
data['Cuisines'] = data['Cuisines'].fillna('Unknown')

# Save original columns for display
original_columns = data[['Cuisines', 'City']]

# Drop irrelevant or textual columns
columns_to_drop = [
    'Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose',
    'Switch to order menu', 'Rating color', 'Rating text'
]
data = data.drop(columns=columns_to_drop, errors='ignore')

# Encode categorical features
categorical_features = ['City', 'Cuisines', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Filter out rows with extreme values or low ratings/votes
data = data[(data['Aggregate rating'] > 2.0) & (data['Votes'] > 10)]
data = data[(data['Average Cost for two'] > 100) & (data['Average Cost for two'] < 50000)]

# Normalize numeric features
scaler = MinMaxScaler()
data[['Average Cost for two', 'Votes', 'Aggregate rating']] = scaler.fit_transform(
    data[['Average Cost for two', 'Votes', 'Aggregate rating']]
)

# User preferences
user_preferences = {
    'Price range': 2,  # Scaled price range
    'Average Cost for two': 0.5,  # Scaled value between 0 and 1
    'Aggregate rating': 0.8,  # Scaled value between 0 and 1
    'Votes': 0.7  # Scaled value between 0 and 1
}

# Add dummy entries for all encoded categorical features in user preferences
for col in data.columns:
    if col not in user_preferences:
        user_preferences[col] = 0

# Convert user preferences to DataFrame
user_pref_df = pd.DataFrame([user_preferences])

# Ensure columns match between the dataset and user preferences
user_pref_df = user_pref_df.reindex(columns=data.columns, fill_value=0)

# Calculate similarity
similarity = cosine_similarity(data.drop(columns=['Similarity'], errors='ignore'), user_pref_df)
data['Similarity'] = similarity

# Sort and get the top recommendations
recommendations = data.sort_values(by='Similarity', ascending=False).head(5)

# Attach original columns for display
recommendations = recommendations.join(original_columns)

# Display additional relevant columns for recommendations
columns_to_display = [
    'Cuisines', 'City', 'Price range', 'Average Cost for two', 
    'Aggregate rating', 'Votes', 'Similarity'
]
recommendations_display = recommendations[columns_to_display]

# Show user-friendly output in the terminal
print("\nTop Restaurant Recommendations:\n")
for index, row in recommendations_display.iterrows():
    print(f"Recommendation {index + 1}:")
    print(f"  - Cuisines: {row['Cuisines']}")
    print(f"  - City: {row['City']}")
    print(f"  - Price Range: {row['Price range']}")
    print(f"  - Average Cost for Two: {row['Average Cost for two']:.2f}")
    print(f"  - Aggregate Rating: {row['Aggregate rating']:.2f}")
    print(f"  - Votes: {row['Votes']:.0f}")
    print(f"  - Similarity Score: {row['Similarity']:.2f}\n")

# Create an interactive scatter plot with multiple attributes
fig = px.scatter(
    recommendations_display,
    x='Similarity',
    y='Aggregate rating',
    size='Votes',  # Bubble size represents the number of votes
    color='City',
    hover_data=['Cuisines', 'Price range', 'Average Cost for two'],
    title='Top Restaurant Recommendations',
    labels={
        'Similarity': 'Similarity Score',
        'Aggregate rating': 'Aggregate Rating',
        'Votes': 'Number of Votes',
        'Average Cost for two': 'Cost for Two (Scaled)'
    }
)

# Customize plot for better interactivity and readability
fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(
    title=dict(font=dict(size=24)),
    xaxis_title='Similarity Score',
    yaxis_title='Aggregate Rating',
    font=dict(size=14),
    legend_title='City'
)

# Show the interactive plot
fig.show()

