import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ====================================
# Load Dataset
# ====================================

df = pd.read_csv("data/heart.csv")

X = df.drop(columns=["target"])
y = df["target"]


# ====================================
# Feature Groups
# ====================================

numeric_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


# ====================================
# Preprocessing
# ====================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ====================================
# Train-Test Split
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ====================================
# Logistic Regression
# ====================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# ====================================
# Random Forest
# ====================================

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


# ====================================
# Train Models
# ====================================

print("====================================")
print("Training Models")
print("====================================")

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)
print("Logistic Regression trained successfully.")

print("\nTraining Random Forest...")
random_forest_model.fit(X_train, y_train)
print("Random Forest trained successfully.")


# ====================================
# Save Models
# ====================================

joblib.dump(
    logistic_model,
    "models/logistic_regression.pkl"
)

joblib.dump(
    random_forest_model,
    "models/random_forest.pkl"
)


print("\n====================================")
print("Models Saved Successfully")
print("====================================")

print("\nSaved files:")
print("models/logistic_regression.pkl")
print("models/random_forest.pkl")