import pandas as pd
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


print("=" * 70)
print("FAKE NEWS DETECTION - MODEL EVALUATION")
print("=" * 70)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

news = pd.read_csv("news_dataset.csv")

X = news["content"]
y = news["label"]

print(f"Total articles: {len(news)}")


# --------------------------------------------------
# 2. CREATE THE SAME TRAIN/TEST SPLIT
# --------------------------------------------------

print("\nCreating test dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training articles: {len(X_train)}")
print(f"Testing articles:  {len(X_test)}")


# --------------------------------------------------
# 3. LOAD TRAINED MODEL AND VECTORIZER
# --------------------------------------------------

print("\nLoading trained model...")

with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

print("Model loaded successfully.")
print("Vectorizer loaded successfully.")


# --------------------------------------------------
# 4. TRANSFORM TEST DATA
# --------------------------------------------------

print("\nConverting text into TF-IDF features...")

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF transformation completed.")


# --------------------------------------------------
# 5. MAKE PREDICTIONS
# --------------------------------------------------

print("\nMaking predictions...")

predictions = model.predict(X_test_tfidf)

print("Predictions completed.")


# --------------------------------------------------
# 6. CALCULATE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1 * 100:.2f}%)")


# --------------------------------------------------
# 7. CLASSIFICATION REPORT
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    predictions,
    target_names=["Fake News", "Real News"]
)

print(report)


# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

print("\nCreating confusion matrix...")

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)


# Create static folder if it doesn't exist
os.makedirs("static", exist_ok=True)


plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["Fake News", "Real News"],
    yticklabels=["Fake News", "Real News"]
)

plt.title("Fake News Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "static/confusion_matrix.png",
    dpi=200
)

plt.close()

print("Saved: static/confusion_matrix.png")


# --------------------------------------------------
# 9. SAVE EVALUATION RESULTS
# --------------------------------------------------

evaluation_results = {
    "model": type(model).__name__,
    "total_articles": len(news),
    "training_articles": len(X_train),
    "testing_articles": len(X_test),
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "confusion_matrix": cm.tolist()
}

with open("evaluation_results.json", "w") as file:
    json.dump(evaluation_results, file, indent=4)


print("\nSaved: evaluation_results.json")


# --------------------------------------------------
# 10. FINAL MESSAGE
# --------------------------------------------------

print("\n" + "=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nGenerated files:")
print("1. static/confusion_matrix.png")
print("2. evaluation_results.json")

print("\nYour model is ready for the next stage.")