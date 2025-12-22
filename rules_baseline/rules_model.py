import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text) #deleting all except a-z and spase
    return text.split()

def keyword_classifier(review):
    words = tokenize(review)
    positive_words = [
        "good", "great", "wonderful", "best", "love", "liked",
        "enjoyed", "excellent", "amazing", "perfect", "well",
        "positive", "nice"
    ]
    negative_words = [
        "negative", "bad", "worst", "boring", "awful", "waste",
        "terrible", "annoying", "poor", "disappointing", "hard"
        ,"pointless","hate"
    ]

    pos_count = 0
    neg_count = 0

    for word in words:
        if word in positive_words:
            pos_count += 1
        elif word in negative_words:
            neg_count += 1

    if pos_count > neg_count:
        return "positive"
    return "negative"

def run_rules_baseline(
    path: str = r"Data\IMDB Dataset.csv",
    text_col: str = "review",
    label_col: str = "sentiment"
):
    print("[rules_model] start")
    print(f"[rules_model] reading: {path}")


    # 1) Load dataset
    df = pd.read_csv(path)  # Convert dataset to pandas DataFrame

    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred=[]
    for r in X_test:
        y_pred.append(keyword_classifier(r))

    acc = accuracy_score(y_test, y_pred)

    print("\n=== RuleClassifier Results ===")
    print(f"Accuracy = {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("[RuleClassifier] end")


if __name__ == "__main__":
    run_rules_baseline()