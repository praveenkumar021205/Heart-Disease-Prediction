
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/heart.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Use the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load tuned Logistic Regression pipeline
model = joblib.load(
    "models/logistic_regression_tuned.pkl"
)

# Transform test features using the fitted preprocessor
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_test_transformed = preprocessor.transform(X_test)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Convert transformed data to a DataFrame
X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)

# Create output directory
Path("visualizations").mkdir(exist_ok=True)

# Explain Logistic Regression predictions
explainer = shap.LinearExplainer(
    classifier,
    X_test_transformed
)

shap_values = explainer(X_test_transformed)

# Global feature importance plot
shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.tight_layout()
plt.savefig(
    "visualizations/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Explain one test example
shap.plots.waterfall(
    shap_values[0],
    max_display=15,
    show=False
)

plt.tight_layout()
plt.savefig(
    "visualizations/shap_patient_explanation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("SHAP analysis completed successfully.")
print("Saved: visualizations/shap_feature_importance.png")
print("Saved: visualizations/shap_patient_explanation.png")
