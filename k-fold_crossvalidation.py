import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('phone_usage_india_ML.csv')

df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

X = df_encoded.drop('Location', axis=1) 
y = df_encoded['Location']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create logistic regression model with max_iter
model = LogisticRegression(max_iter=500)

# 5-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_scaled, y, cv=kfold)

print("Cross-validation scores:", scores)
