import pandas as pd
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print("=" * 70)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 70)


# ---------------------------------------------------------
# 1. LOAD PREPARED DATASET
# ---------------------------------------------------------

print("\nLoading dataset...")

news = pd.read_csv("news_dataset.csv")

print(f"Total articles: {len(news)}")


# ---------------------------------------------------------
# 2. SEPARATE FEATURES AND LABEL
# ---------------------------------------------------------

X = news["content"]
y = news["label"]


# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training articles: {len(X_train)}")
print(f"Testing articles:  {len(X_test)}")


# ---------------------------------------------------------
# 4. TF-IDF TEXT VECTORIZATION
# ---------------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Training feature matrix: {X_train_tfidf.shape}")
print(f"Testing feature matrix:  {X_test_tfidf.shape}")


# ---------------------------------------------------------
# 5. DEFINE MODELS
# ---------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB()
}


# ---------------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# ---------------------------------------------------------

results = {}

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


# ---------------------------------------------------------
# 7. SELECT BEST MODEL
# ---------------------------------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = models[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Selected model: {best_model_name}")
print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")


# ---------------------------------------------------------
# 8. SAVE MODEL + VECTORIZER
# ---------------------------------------------------------

print("\nSaving trained model...")

with open("fake_news_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


# ---------------------------------------------------------
# 9. SAVE MODEL RESULTS
# ---------------------------------------------------------

training_info = {
    "total_articles": len(news),
    "training_articles": len(X_train),
    "testing_articles": len(X_test),
    "best_model": best_model_name,
    "results": results
}

with open("model_results.json", "w") as file:
    json.dump(training_info, file, indent=4)


print("\nFiles created:")
print("fake_news_model.pkl")
print("tfidf_vectorizer.pkl")
print("model_results.json")

print("\nTraining completed successfully!")

print("=" * 70)