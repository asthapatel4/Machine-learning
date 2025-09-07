import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("phone_usage_india_ML.csv")


print("Column Names:", df.columns.tolist())

target_column = 'Primary Use'

if target_column not in df.columns:
    raise ValueError(f"Column '{target_column}' not found in dataset.")

# Manual encoding
class_mapping = {label: idx for idx, label in enumerate(np.unique(df[target_column]))}
df[target_column + "_mapped"] = df[target_column].map(class_mapping)

# Reverse mapping
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df[target_column + "_original"] = df[target_column + "_mapped"].map(inv_class_mapping)

# LabelEncoder
le = LabelEncoder()
df['encoded_by_labelencoder'] = le.fit_transform(df[target_column])


print(df[[target_column, target_column + "_mapped", target_column + "_original", 'encoded_by_labelencoder']].head())
