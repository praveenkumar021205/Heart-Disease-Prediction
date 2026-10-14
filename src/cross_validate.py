
import pandas as pd
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

# Load dataset
df = pd.read_csv("data/heart.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Load saved models
models = {
    "Logistic Regression": joblib.load(
        "models/logistic_regression.pkl"
    ),
    "Random Forest": joblib.load(
        "models/random_forest.pkl"
    )
}

# Stratified 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print("\n====================================")
    print(name)
    print("5-Fold Cross-Validation")
    print("====================================")

    for metric in scoring:
        values = scores[f"test_{metric}"]
        print(
            f"{metric}: "
            f"{values.mean():.4f} "
            f"+/- {values.std():.4f}"
        )

print("\nCross-validation completed.")
