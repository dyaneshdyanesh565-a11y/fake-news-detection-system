import pandas as pd

# Load the datasets
fake_news = pd.read_csv("Fake.csv")
real_news = pd.read_csv("True.csv")

print("=" * 60)
print("FAKE NEWS DATASET")
print("=" * 60)

print("Number of rows:", len(fake_news))
print("Columns:", list(fake_news.columns))

print("\nFirst 5 fake news articles:")
print(fake_news.head())

print("\nMissing values:")
print(fake_news.isnull().sum())


print("\n" + "=" * 60)
print("REAL NEWS DATASET")
print("=" * 60)

print("Number of rows:", len(real_news))
print("Columns:", list(real_news.columns))

print("\nFirst 5 real news articles:")
print(real_news.head())

print("\nMissing values:")
print(real_news.isnull().sum())


print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Fake news articles:", len(fake_news))
print("Real news articles:", len(real_news))
print("Total articles:", len(fake_news) + len(real_news))