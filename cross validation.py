import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier # A simple model
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Your Data ---
# Make sure 'MultipleFiles/phone_usage_india_ML.csv' is in the correct path
df = pd.read_csv('phone_usage_india_ML.csv')

# --- 2. Prepare Data (Simplified) ---
# We'll use 'Age' as a simple feature and 'Primary Use' as the target.
# For simplicity, we'll just drop other columns and handle categorical data.

# Drop 'User ID' as it's not a feature
df = df.drop('User ID', axis=1)

# Separate features (X) and target (y)
X = df[['Age']] # Using only 'Age' as a feature for simplicity
y = df['Primary Use']

# Convert 'Primary Use' (target) into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 3. Create a Simple Model ---
# A Decision Tree Classifier is easy to understand and use.
model = DecisionTreeClassifier(random_state=42) # random_state for consistent results

# --- 4. Perform Cross-Validation ---
# This function automatically handles splitting the data into folds,
# training the model on some folds, and testing on others.
# cv=5 means it will do this 5 times (5-fold cross-validation).
scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')

# --- 5. Show Results ---
print("Scores for each fold (accuracy):", scores)
print("Average accuracy across all folds:", scores.mean())
print("How much the scores varied (standard deviation):", scores.std())

print("\n--- What does this mean? ---")
print(f"Your model achieved an average accuracy of {scores.mean():.2f} across 5 different tests.")
print("This gives you a more reliable idea of how well your model performs,")
print("compared to just testing it once on a single split of your data.")
