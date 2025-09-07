import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/mnt/data/phone_usage_india_ML.csv")

# Target variable
y = df['Primary Use']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Features (drop ID + target)
X = df.drop(columns=['User ID', 'Primary Use'])
X = pd.get_dummies(X, columns=['Location', 'Phone Brand'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42, max_depth=4)  # limit depth for better visualization
dt.fit(X_train, y_train)

# Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=label_encoder.classes_, filled=True, rounded=True, fontsize=10)
tree_plot_path = "/mnt/data/decision_tree.png"
plt.savefig(tree_plot_path, dpi=300, bbox_inches="tight")
plt.close()

tree_plot_path
