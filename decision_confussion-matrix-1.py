import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("D:/MU/sem-3/ML_project/phone_usage_india_ML.csv")
df.drop(columns=["User ID"], inplace=True)

# Encode the target column
target_col = "Primary Use"
le = LabelEncoder()
df["Primary Use Encoded"] = le.fit_transform(df[target_col])

# Drop original target after encoding
y = df["Primary Use Encoded"]

X = df.drop(columns=["Primary Use", "Primary Use Encoded"])
X = pd.get_dummies(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train decision tree
dtc = DecisionTreeClassifier(random_state=40)
dtc.fit(x_train, y_train)

y_pred = dtc.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
