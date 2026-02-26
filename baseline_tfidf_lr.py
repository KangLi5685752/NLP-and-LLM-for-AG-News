from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import numpy as np

def main():
    print("Loading dataset...")
    ds = load_dataset("ag_news")

    X_train = ds["train"]["text"]
    y_train = ds["train"]["label"]
    X_test = ds["test"]["text"]
    y_test = ds["test"]["label"]

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        ngram_range=(1,2)
    )

    start_time = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=200, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)

    # =============================
    # Feature Importance Analysis
    # =============================

    print("\nTop important words for each class:")

    feature_names = vectorizer.get_feature_names_out()
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    # Logistic Regression gives coefficients for each class
    for i, class_label in enumerate(class_names):
        coef = model.coef_[i]
    
        # Get top 10 features with highest weights
        top_indices = np.argsort(coef)[-10:]
        top_words = [feature_names[j] for j in top_indices]
    
        print(f"\nClass: {class_label}")
        print(", ".join(top_words))

    print("Predicting...")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    elapsed = time.time() - start_time

    print("\n===== Baseline Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(f"Training + inference time: {elapsed:.2f} seconds")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()