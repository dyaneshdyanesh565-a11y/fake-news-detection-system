import pandas as pd

print("=" * 60)
print("FAKE NEWS DATASET PREPARATION")
print("=" * 60)

# Load datasets
fake_news = pd.read_csv("Fake.csv")
real_news = pd.read_csv("True.csv")

print(f"\nOriginal fake articles: {len(fake_news)}")
print(f"Original real articles: {len(real_news)}")


# Add labels
fake_news["label"] = 0
real_news["label"] = 1


# Combine both datasets
news = pd.concat([fake_news, real_news], ignore_index=True)


# Combine title and article text
news["title"] = news["title"].fillna("")
news["text"] = news["text"].fillna("")

news["content"] = news["title"] + " " + news["text"]


# Remove empty articles
news = news[news["content"].str.strip() != ""]


# Remove duplicate articles
before_duplicates = len(news)

news = news.drop_duplicates(subset=["content"])

duplicates_removed = before_duplicates - len(news)


# Keep only the columns we need
news = news[["content", "label"]]


# Shuffle the dataset
news = news.sample(frac=1, random_state=42).reset_index(drop=True)


# Save prepared dataset
news.to_csv("news_dataset.csv", index=False)


print("\n" + "=" * 60)
print("PREPARATION COMPLETE")
print("=" * 60)

print(f"Duplicates removed: {duplicates_removed}")
print(f"Final articles: {len(news)}")

print("\nLabel distribution:")
print(news["label"].value_counts())

print("\nLabel meaning:")
print("0 = Fake News")
print("1 = Real News")

print("\nSaved file:")
print("news_dataset.csv")

print("\nFirst 3 records:")
print(news.head(3))