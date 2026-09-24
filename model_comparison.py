import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# MODEL COMPARISON - ACADEMIC PERFORMANCE PREDICTION
# ============================================================

results = pd.DataFrame({
    "Model": [
        "ANN Baseline",
        "Multi-Modal LSTM",
        "Hybrid LSTM-Transformer"
    ],
    "Accuracy": [0.7681, 0.7177, 0.776805],
    "Precision": [0.7679, 0.7405, 0.778219],
    "Recall": [0.7681, 0.7177, 0.776805],
    "F1_Score": [0.7675, 0.7195, 0.776230]
})

print("=" * 72)
print("MODEL COMPARISON")
print("=" * 72)

display_table = results.copy()

for col in ["Accuracy", "Precision", "Recall", "F1_Score"]:
    display_table[col] = (display_table[col] * 100).round(2)

print("\nPerformance Comparison (%):")
print(display_table.to_string(index=False))

# Save comparison table
results.to_csv(
    "model_comparison_results.csv",
    index=False
)

# ------------------------------------------------------------
# COMPARISON GRAPH
# ------------------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title("Performance Comparison of Prediction Models")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)

plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

best_index = results["Accuracy"].idxmax()
best_model = results.loc[best_index, "Model"]
best_accuracy = results.loc[best_index, "Accuracy"]

print("\n" + "=" * 72)
print("BEST MODEL")
print("=" * 72)

print(f"Model    : {best_model}")
print(f"Accuracy : {best_accuracy * 100:.2f}%")

print("\nSaved:")
print(" - model_comparison_results.csv")
print(" - model_comparison.png")

print("\n" + "=" * 72)
print("MODEL COMPARISON COMPLETED")
print("=" * 72)
