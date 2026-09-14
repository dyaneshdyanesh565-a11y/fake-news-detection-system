import json
import os
import matplotlib.pyplot as plt

print("=" * 70)
print("CREATING MODEL COMPARISON CHART")
print("=" * 70)

print("\nLoading model_results.json...")

with open("model_results.json", "r") as file:
    data = json.load(file)

results = data["results"]

models = list(results.keys())

accuracy = [
    results[model]["accuracy"] * 100
    for model in models
]

os.makedirs("static", exist_ok=True)

# Create chart
plt.figure(figsize=(10, 6))

bars = plt.bar(
    models,
    accuracy,
    color="#e50914"
)

plt.title(
    "Fake News Detection - Model Accuracy Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Machine Learning Model",
    fontsize=12
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=12
)

plt.ylim(0, 105)

# Display accuracy values above each bar
for bar, value in zip(bars, accuracy):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.xticks(rotation=10)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

chart_path = "static/model_comparison.png"

plt.savefig(
    chart_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nChart created successfully!")

print("\nSaved file:")
print(chart_path)

print("\n" + "=" * 70)
print("MODEL COMPARISON CHART COMPLETED")
print("=" * 70)