
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/heart.csv")
X = df.drop(columns=["target"])
y = df["target"]

numeric_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

# Preprocessing
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

# Hold out the test set until final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Logistic Regression tuning
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

lr_params = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"]
}

lr_search = GridSearchCV(
    lr_pipeline,
    lr_params,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

# Random Forest tuning
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42
    ))
])

rf_params = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__class_weight": [None, "balanced"]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

# Fit on training data only
print("Tuning Logistic Regression...")
lr_search.fit(X_train, y_train)

print("\nTuning Random Forest...")
rf_search.fit(X_train, y_train)

# Save tuned pipelines
joblib.dump(
    lr_search.best_estimator_,
    "models/logistic_regression_tuned.pkl"
)

joblib.dump(
    rf_search.best_estimator_,
    "models/random_forest_tuned.pkl"
)

print("\n====================================")
print("HYPERPARAMETER TUNING RESULTS")
print("====================================")

print("\nLogistic Regression best parameters:")
print(lr_search.best_params_)
print("Best CV ROC-AUC:", round(lr_search.best_score_, 4))

print("\nRandom Forest best parameters:")
print(rf_search.best_params_)
print("Best CV ROC-AUC:", round(rf_search.best_score_, 4))

print("\nTuned models saved successfully.")
print("Test data has not been used for tuning.")
