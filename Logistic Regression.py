# Imports
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("phone_usage_india_ML.csv")

# 🎯 Target variable
y = df['Primary Use']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 📊 Feature selection: Drop User ID & Primary Use (target)
X = df.drop(columns=['User ID', 'Primary Use'])

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['Location', 'Phone Brand'], drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# 3-Fold Cross-Validation
scores_3 = cross_val_score(logreg, X_scaled, y_encoded, cv=3)
print("Three-fold CV scores:", scores_3)
print("Average 3-fold CV score: {:.2f}".format(scores_3.mean()))

# 5-Fold Cross-Validation
scores_5 = cross_val_score(logreg, X_scaled, y_encoded, cv=5)
print("Five-fold CV scores:", scores_5)
print("Average 5-fold CV score: {:.2f}".format(scores_5.mean()))
