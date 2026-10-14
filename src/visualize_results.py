
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Actual evaluation results
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [0.8852, 0.8525],
    "Precision": [0.8387, 0.8065],
    "Recall": [0.9286, 0.8929],
    "F1 Score": [0.8814, 0.8475],
    "ROC-AUC": [0.9665, 0.9432]
})

# Create output folders
Path("visualizations").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Save metrics
results.to_csv("data/model_metrics.csv", index=False)

# Plot comparison
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=0
)

ax.set_title("Heart Disease Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "visualizations/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Evaluation metrics saved to data/model_metrics.csv")
print("Comparison chart saved to visualizations/model_comparison.png")
