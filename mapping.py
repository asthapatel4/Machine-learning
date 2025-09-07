import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("phone_usage_india_ML.csv")


categories = df['Primary Use'].values

# LabelEncoder
le = LabelEncoder()
encoded = le.fit_transform(categories)

# Decode back to original labels
decoded = le.inverse_transform(encoded)


df['PrimaryUse_Encoded'] = encoded
df['PrimaryUse_Decoded'] = decoded

print("Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(df[['Primary Use', 'PrimaryUse_Encoded', 'PrimaryUse_Decoded']].head())
