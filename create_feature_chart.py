import json
import pickle
import os
import matplotlib.pyplot as plt


print("=" * 70)
print("CREATING FEATURE IMPORTANCE CHART")
print("=" * 70)


# Load trained model
print("\nLoading trained model...")

with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully.")


# Load TF-IDF vectorizer
print("\nLoading TF-IDF vectorizer...")

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

print("Vectorizer loaded successfully.")


# Check whether model has coefficients
if not hasattr(model, "coef_"):

    print("\nERROR:")
    print("The selected model does not provide feature coefficients.")
    print("Feature importance chart cannot be created.")

else:

    print("\nExtracting feature importance...")

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get model coefficients
    coefficients = model.coef_[0]

    # Find 15 most important features for each direction
    top_fake_indices = coefficients.argsort()[:15]
    top_real_indices = coefficients.argsort()[-15:][::-1]

    fake_features = [
        feature_names[index]
        for index in top_fake_indices
    ]

    fake_values = [
        abs(coefficients[index])
        for index in top_fake_indices
    ]

    real_features = [
        feature_names[index]
        for index in top_real_indices
    ]

    real_values = [
        coefficients[index]
        for index in top_real_indices
    ]

    # Combine features
    features = fake_features[::-1] + real_features
    values = [-value for value in fake_values[::-1]] + real_values

    # Create static folder if needed
    os.makedirs("static", exist_ok=True)

    # Create chart
    plt.figure(figsize=(12, 10))

    plt.barh(
        features,
        values,
        color="#e50914"
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.title(
        "Fake News Detection - Influential TF-IDF Features",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Feature Influence",
        fontsize=12
    )

    plt.ylabel(
        "Words / Phrases",
        fontsize=12
    )

    plt.tight_layout()

    chart_path = "static/feature_importance.png"

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
    print("FEATURE IMPORTANCE CHART COMPLETED")
    print("=" * 70)