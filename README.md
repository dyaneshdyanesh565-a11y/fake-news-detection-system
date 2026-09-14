# 🛡️ NewsGuard AI — Fake News Detection System

A machine-learning-based web application that analyzes news content and predicts whether it resembles **Fake News** or **Real News** using Natural Language Processing (NLP).

---

## 📌 Project Overview

NewsGuard AI is a Fake News Detection and News Credibility Analysis System developed using Python, Flask, NLP, and Machine Learning.

The system accepts a news headline or article as input and analyzes its linguistic patterns using **TF-IDF vectorization** and a **Linear Support Vector Machine (SVM)** classifier.

It also provides:

- Fake/Real news prediction
- Model confidence level
- Decision score
- Text statistics
- Credibility-style indicators
- Influential words/features
- Model comparison
- Confusion matrix
- Classification metrics
- PDF analysis report

> ⚠️ This system identifies patterns learned from its training dataset. It does not independently verify facts, sources, people, events, or claims.

---

## ✨ Features

### 📰 News Analysis
- Enter a news headline or complete article.
- Analyze the submitted text using the trained ML model.
- Receive a Fake News or Real News prediction.

### 🤖 Machine Learning
The project compares three machine-learning algorithms:

1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes

The model with the best test F1 score is selected.

### 🔎 NLP Processing

The system uses:

- Text preprocessing
- TF-IDF vectorization
- Unigram and bigram features
- Stop-word removal
- Feature analysis

### 📊 Text Analysis

The application calculates:

- Word count
- Character count
- Sentence count
- Average sentence length
- Uppercase word count
- Exclamation mark count

### 🚨 Credibility Indicators

The system checks for basic linguistic patterns such as:

- Sensational wording
- Excessive punctuation
- Clickbait-style phrases
- Suspicious linguistic patterns

These are indicators only and are not proof that an article is fake.

### 🔬 Influential Features

The application displays words and phrases that had stronger influence on the machine-learning model's prediction.

### 📈 Model Evaluation

The project includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Model Comparison Chart
- Feature Importance Chart

### 📄 PDF Report

Users can generate a professional PDF analysis report containing the news analysis and prediction results.

---

## 🧠 System Architecture

```text
                USER
                  │
                  ▼
          Flask Web Application
                  │
                  ▼
          News Article Input
                  │
                  ▼
         Text Preprocessing
                  │
                  ▼
          TF-IDF Vectorizer
                  │
                  ▼
            Linear SVM
                  │
                  ▼
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
   Prediction          Decision Score
       │                     │
       └──────────┬──────────┘
                  ▼
        Credibility Analysis
                  │
                  ▼
          Result Dashboard
                  │
                  ▼
             PDF Report