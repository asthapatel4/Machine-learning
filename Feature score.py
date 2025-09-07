#Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv("phone_usage_india_ML.csv")

#Drop non-useful columns
df_cleaned = df.drop(columns=["User ID"])  # ID is not predictive

#Encode categorical features
label_encoders = {}
for col in df_cleaned.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

#Split into features and target
X = df_cleaned.drop(columns=["Primary Use"])
y = df_cleaned["Primary Use"]

#Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

#Get and sort feature importances
feature_scores = pd.Series(clf.feature_importances_, index=X.columns)
feature_scores_sorted = feature_scores.sort_values(ascending=False)

#Plot feature importances
plt.figure(figsize=(10, 6))
feature_scores_sorted.plot(kind='bar', color='skyblue')
plt.title("Feature Importance using Decision Tree")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.tight_layout()
plt.grid(True)
plt.show()

#Print feature scores
print("Feature Importance Scores (Descending Order):\n")
print(feature_scores_sorted)
