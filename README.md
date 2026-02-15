# Sports vs Politics Text Classification

## 📌 Project Overview

This project focuses on building a supervised machine learning classifier that categorizes text documents into one of two domains:

- **Sports**
- **Politics**

The objective was to:
1. Construct a large-scale labeled dataset.
2. Compare multiple feature representation techniques.
3. Evaluate at least three machine learning models.
4. Perform quantitative comparison using standard evaluation metrics.
5. Deploy the best-performing model for real-time classification.

---

# Problem Statement

Design a classifier that reads a text document and classifies it as:

- **Sports**
- **Politics**

You may use:
- Bag of Words
- TF-IDF
- N-grams

Compare at least three ML techniques and provide quantitative analysis.

---

#  Dataset Construction

##  Data Source

The dataset was collected using automated web crawling from **Wikipedia**.

Two primary domain category pages were used:

- https://en.wikipedia.org/wiki/Category:Sports
- https://en.wikipedia.org/wiki/Category:Politics

---

##  Keyword Generation Strategy

Instead of manually creating keywords, domain-specific keywords were automatically extracted from Wikipedia category pages.

Up to 500 keywords were generated per domain from article titles listed in the category.

This ensured:

- Domain relevance
- Reproducibility
- No manual bias
- Structured taxonomy usage

---

##  Title-Based Filtering

To prevent topic drift:

- Only articles whose titles contained at least one domain keyword were processed.
- Disambiguation pages were excluded.
- Meta pages (containing “:”) were excluded.

This strict filtering ensured domain consistency.

---

##  Sentence Extraction

For each valid article:

1. Paragraph text was extracted.
2. Text was tokenized into sentences using NLTK.
3. Sentences shorter than 8 words were discarded.

---

##  Final Dataset Statistics

- Total Sentences: **50,000**
- Sports: 25,000
- Politics: 25,000
- Fully balanced dataset

Each sample contains:
- A sentence
- A binary label (0 = Sports, 1 = Politics)

---

#  Feature Representation Techniques

Three feature extraction techniques were compared:

## 1️⃣ Bag of Words (BoW)

Represents text as word frequency counts.
Ignores word order.

---

## 2️⃣ TF-IDF

Term Frequency – Inverse Document Frequency.
Reduces impact of common words.
Highlights discriminative terms.

---

## 3️⃣ N-grams (Unigram + Bigram)

Captures contextual word sequences.
Example:
- "world cup"
- "prime minister"
- "public policy"

---

#  Machine Learning Models Compared

Four classifiers were evaluated:

1. **Naive Bayes**
2. **Logistic Regression**
3. **Linear SVM**
4. **Random Forest**

---

#  Experimental Setup

- 80% Training Data
- 20% Testing Data
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

---

#  Results Summary

## 🔹 Feature Comparison (Using Naive Bayes)

| Feature | Accuracy |
|----------|----------|
| Bag of Words | 96.39% |
| TF-IDF | 95.85% |
| **N-grams (1,2)** | **96.59%** |

N-grams achieved the best performance due to contextual representation.

---

## 🔹 Model Comparison (Using N-grams)

| Model | Accuracy |
|--------|----------|
| **Naive Bayes** | **96.59%** |
| Logistic Regression | 96.38% |
| Linear SVM | 96.17% |
| Random Forest | 94.44% |

---

#  Best Configuration

- Feature: **N-grams (1,2)**
- Model: **Naive Bayes**
- Accuracy: **96.59%**

---
# ▶ How to Run the Project
Follow the steps below to execute the complete project pipeline.

## Step 1: Generate the Dataset
First, run the dataset generation script:
```bash
python Generate_Dataset.py
```
This script will crawl Wikipedia, apply strict title-based filtering, extract relevant sentences, and generate the dataset file (dataset_title_filtered.csv). Ensure the dataset file is successfully created before proceeding.

## Step 2: Run the Classifier

After the dataset has been generated, execute:
```bash
python classifier.py
```

This script will train the best-performing model (if not already saved), load the trained model using joblib, and allow real-time classification of user input text.

You can enter any sentence, and the model will classify it as:

SPORTS
or
POLITICS


