import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Setup Logging
# ----------------------------

logging.basicConfig(
    filename="experiment_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("dataset_title_filtered.csv")

X = df["sentence"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Utility Function
# ----------------------------

def evaluate_model(model, X_train_vec, X_test_vec, name):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"{name} -> Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")

    return acc, prec, rec, f1, y_pred


# ==========================================================
# PART 1: Compare Feature Representations (Using Naive Bayes)
# ==========================================================

feature_methods = {
    "Bag_of_Words": CountVectorizer(),
    "TF-IDF": TfidfVectorizer(),
    "N-grams (1,2)": CountVectorizer(ngram_range=(1,2))
}

feature_results = {}

for name, vectorizer in feature_methods.items():
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    acc, prec, rec, f1, _ = evaluate_model(model, X_train_vec, X_test_vec, name)

    feature_results[name] = acc

# Plot Feature Comparison
plt.figure(figsize=(8,5))
plt.bar(feature_results.keys(), feature_results.values())
plt.title("Feature Representation Comparison (Naive Bayes)")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("feature_comparison.png")
plt.close()

best_feature = max(feature_results, key=feature_results.get)
logging.info(f"Best Feature Technique: {best_feature}")

# ==========================================================
# PART 2: Compare 4 ML Models Using Best Feature
# ==========================================================

if best_feature == "Bag_of_Words":
    best_vectorizer = CountVectorizer()
elif best_feature == "TF-IDF":
    best_vectorizer = TfidfVectorizer()
else:
    best_vectorizer = CountVectorizer(ngram_range=(1,2))

X_train_vec = best_vectorizer.fit_transform(X_train)
X_test_vec = best_vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

model_results = {}
conf_matrices = {}

for name, model in models.items():
    acc, prec, rec, f1, y_pred = evaluate_model(model, X_train_vec, X_test_vec, name)
    model_results[name] = acc
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

# Plot Model Comparison
plt.figure(figsize=(8,5))
plt.bar(model_results.keys(), model_results.values())
plt.title(f"Model Comparison Using {best_feature}")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# Plot Confusion Matrices
for name, cm in conf_matrices.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_{name.replace(' ','_')}.png")
    plt.close()

print("Experiment completed successfully!")
print("Check experiment_log.txt and generated plots.")
