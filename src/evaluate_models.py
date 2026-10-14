
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/heart.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Use the same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load saved models
logistic_model = joblib.load(
    "models/logistic_regression.pkl"
)
random_forest_model = joblib.load(
    "models/random_forest.pkl"
)

# Evaluation function
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n====================================")
    print(name)
    print("====================================")

    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# Evaluate both models
evaluate_model(
    "Logistic Regression",
    logistic_model,
    X_test,
    y_test
)

evaluate_model(
    "Random Forest",
    random_forest_model,
    X_test,
    y_test
)

print("\nEvaluation completed successfully.")
