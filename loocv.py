# Imports
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("D:/MU/sem-3/ML_project/phone_usage_india_ML.csv")

# Drop 'User  ID' as it's not a feature
df = df.drop('User  ID', axis=1)

# Separate features (X) and target (y)
X = df.drop('Primary Use', axis=1)  # Features
y = df['Primary Use']  # Target variable

# Encode categorical target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create the Logistic Regression model
logreg = LogisticRegression(max_iter=200)

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
scores = cross_val_score(logreg, X, y_encoded, cv=loo)

# Print results
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
