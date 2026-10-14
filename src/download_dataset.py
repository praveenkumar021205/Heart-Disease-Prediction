from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Convert target into binary classification
# 0 = No Heart Disease
# 1,2,3,4 = Heart Disease
df["target"] = (df["num"] > 0).astype(int)

# Remove original target column
df = df.drop(columns=["num"])

# Save cleaned dataset
df.to_csv("data/heart.csv", index=False)

print("====================================")
print("Heart Disease Dataset Created")
print("====================================")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved file:")
print("data/heart.csv")