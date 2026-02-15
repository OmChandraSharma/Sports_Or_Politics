import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_FILE = "sports_politics_model.joblib"
VECTORIZER_FILE = "vectorizer.joblib"
DATASET_FILE = "dataset_title_filtered.csv"

# ---------------------------------------------------
# TRAIN AND SAVE MODEL (if not already saved)
# ---------------------------------------------------

def train_and_save():
    print("Training model...")

    df = pd.read_csv(DATASET_FILE)

    X = df["sentence"]
    y = df["label"]

    # Best Feature: N-grams (1,2)
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X_vectorized = vectorizer.fit_transform(X)

    # Best Model: Naive Bayes
    model = MultinomialNB()
    model.fit(X_vectorized, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("Model trained and saved successfully!\n")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

def load_model():
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return model, vectorizer

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    # If model files don't exist, train first
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        train_and_save()

    model, vectorizer = load_model()

    print("Model ready for classification!")
    print("Type a sentence (or type 'exit' to quit)\n")

    while True:
        user_input = input("Enter text: ")

        if user_input.lower() == "exit":
            print("Exiting...")
            break

        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]

        if prediction == 0:
            label = "SPORTS 🏆"
            confidence = probabilities[0]
        else:
            label = "POLITICS 🏛"
            confidence = probabilities[1]

        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}\n")


if __name__ == "__main__":
    main()
