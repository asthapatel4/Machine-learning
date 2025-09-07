import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = 'phone_usage_india_ML.csv' # Corrected file path based on previous context
df = pd.read_csv(file_path)

# Data Preparation
# Identify features (X) and target (y)
# We'll use numerical features for X and 'Primary Use' as y
numerical_features = [
    'Age', 'Screen Time (hrs/day)', 'Data Usage (GB/month)',
    'Calls Duration (mins/day)', 'Number of Apps Installed',
    'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
    'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)',
    'Monthly Recharge Cost (INR)'
]
X = df[numerical_features]
y = df['Primary Use']

# Encode the target variable 'Primary Use'
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale the entire dataset X before splitting for PCA consistency
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # <--- FIX: Scale the entire X here

# Split the data into training and testing sets using the scaled X
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Note: X_train_scaled and X_test_scaled are now simply X_train and X_test
# from the split of the already scaled X_scaled.
# So, you can directly use X_train and X_test in the SVC model training.

# Support Vector Classification (SVC) Model
print("Training Support Vector Classifier...")
svc_model = SVC(kernel='linear', random_state=42)
svc_model.fit(X_train, y_train) # Use X_train (which is already scaled)
print("SVC training complete.")

# Model Evaluation
y_pred = svc_model.predict(X_test) # Use X_test (which is already scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# SVM Graph Generation
# Reduce dimensionality to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled) # Now X_scaled is defined

# Re-split the PCA transformed data
# It's good practice to re-split after PCA if you want to train a new model
# specifically for the 2D visualization, ensuring the train/test sets are consistent.
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Train SVC on 2D PCA-transformed data for visualization
svc_model_2d = SVC(kernel='linear', random_state=42)
svc_model_2d.fit(X_train_pca, y_train_pca)

# Plotting the SVM decision boundary
plt.figure(figsize=(10, 8))

# Create a mesh to plot in
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot decision boundary
Z = svc_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

# Plot data points
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=80, alpha=0.8, edgecolor='k')

# Plot support vectors
plt.scatter(svc_model_2d.support_vectors_[:, 0], svc_model_2d.support_vectors_[:, 1], s=100,
            facecolors='none', edgecolors='red', linewidths=1.5, label='Support Vectors')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Decision Boundary (2D PCA)')
plt.legend(title='Primary Use')
plt.grid(True)
plt.show()
