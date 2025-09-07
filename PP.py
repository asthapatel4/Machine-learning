import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------- 1. Load Dataset -------------------
df = pd.read_csv("phone_usage_india_ML.csv")
df.drop(columns=["User ID"], inplace=True)

# ------------------- 2. Encode Target -------------------
target = "Primary Use"
le_target = LabelEncoder()
df[target] = le_target.fit_transform(df[target])

# ------------------- 3. Feature Engineering -------------------
df["Total Entertainment Time"] = (
    df["Streaming Time (hrs/day)"] + df["Gaming Time (hrs/day)"] + df["Social Media Time (hrs/day)"]
)

df["Avg Usage per App"] = df["Screen Time (hrs/day)"] / df["Number of Apps Installed"]
df["Recharge per GB"] = df["Monthly Recharge Cost (INR)"] / df["Data Usage (GB/month)"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ------------------- 4. Separate Features & Target -------------------
X = df.drop(columns=[target])
y = df[target]

# One-Hot Encoding categorical columns
categorical_cols = ["Location", "Phone Brand"]
ct = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)], remainder="passthrough")
X_encoded = ct.fit_transform(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# ------------------- 5. Train-Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ------------------- 6. Model Training (Logistic Regression with class balance) -------------------
log_reg = LogisticRegression(max_iter=500, class_weight="balanced")
log_reg.fit(X_train, y_train)

# Predictions
y_pred_prob = log_reg.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# ------------------- 7. Evaluation -------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=0))

# ------------------- 8. Cross-Validation -------------------
kf = KFold(n_splits=5, shuffle=True, random_state=0)
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=kf)
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# ------------------- 9. Graphs -------------------
# Line Graph: Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100], label="Actual", linestyle='-', color='blue')
plt.plot(y_pred[:100], label="Predicted", linestyle='--', color='red')
plt.title("Logistic Regression: Actual vs Predicted (first 100 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.legend()
plt.grid(True)
plt.show()

# Dotted Graph: Predicted Probabilities for first 3 classes
plt.figure(figsize=(12,6))
for i in range(min(3, y_pred_prob.shape[1])):
    plt.plot(y_pred_prob[:100, i], linestyle=':', marker='o', label=f"Class {i} Probability")
plt.title("Logistic Regression Predicted Probabilities (first 100 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()
