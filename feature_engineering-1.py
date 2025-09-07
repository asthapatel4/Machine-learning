import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# dataset
df = pd.read_csv("phone_usage_india.csv")

print(" Original Columns:", df.columns.tolist())


df.drop(columns=["User ID"], inplace=True)

# 3. Encode categorical columns
categorical_cols = ['Gender', 'Location', 'Phone Brand', 'OS', 'Primary Use']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    label_encoders[col] = le


df['Total Entertainment Time (hrs/day)'] = (
    df['Streaming Time (hrs/day)'] + 
    df['Gaming Time (hrs/day)'] +
    df['Social Media Time (hrs/day)']
)

df['Avg Usage per App (hrs/day)'] = df['Screen Time (hrs/day)'] / df['Number of Apps Installed']
df['Recharge per GB'] = df['Monthly Recharge Cost (INR)'] / df['Data Usage (GB/month)']


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


scaler = StandardScaler()
numerical_cols = ['Total Entertainment Time (hrs/day)', 'Avg Usage per App (hrs/day)', 'Recharge per GB']


df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=[col + "_scaled" for col in numerical_cols])

df_final = pd.concat([df, df_scaled], axis=1)


print("\n Engineered Feature Values (Unscaled):")
print(df[['Total Entertainment Time (hrs/day)', 
          'Avg Usage per App (hrs/day)', 
          'Recharge per GB']].head())



print("\n Final Engineered Dataset Sample:")
print(df_final.head())
