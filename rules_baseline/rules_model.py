import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Tokenizes a text review:
# 1) Converts text to lowercase
# 2) Removes all characters except lowercase letters and spaces
# 3) Splits the text into individual words
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text) #deleting all except a-z and space
    return text.split()
#"good", "nice" ,
#"hard","hate",
# The review is first tokenized into a list of words.
def keyword_classifier(review):
    words = tokenize(review)
    positive_words = [
         "great", "best", "love", "liked",
         "perfect", "well","positive"
        , "wonderful", "excellent", "amazing", "enjoyed"
    ]
    negative_words = [
        "bad","negative","worst", "boring", "awful", "waste",
        "pointless", "terrible", "annoying", "poor", "disappointing"
    ]

    pos_count = 0
    neg_count = 0
# Count how many positive and negative keywords appear in the review.
    for word in words:
        if word in positive_words:
            pos_count += 1
        elif word in negative_words:
            neg_count += 1

    # Classification rule:
    # If the number of positive words is greater than the number of negative words,
    # classify the review as positive; otherwise classify it as negative.
    if pos_count > neg_count:
        return "positive"
    return "negative"

# Runs the rule-based baseline model on the IMDB dataset.
# This function loads the data, applies the keyword classifier,
# and evaluates the results.
def run_rules_baseline(
    path: str = r"Data\IMDB Dataset.csv",
    text_col: str = "review",
    label_col: str = "sentiment"
):
    print("[rules_model] start")
    print(f"[rules_model] reading: {path}")


    # 1) Load dataset
    df = pd.read_csv(path)  # Convert dataset to pandas DataFrame

    # Extract the review texts (features) and sentiment labels.
    # Both are converted to strings to ensure consistent processing.
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    # Split the dataset into training and test sets.
    # 20% of the data is used for testing.
    # Stratification ensures the class distribution is preserved.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred=[]
    # Apply the rule-based classifier to each review in the test set
    # and store the predicted labels.
    for r in X_test:
        y_pred.append(keyword_classifier(r))

    # Compute the classification accuracy on the test set.
    acc = accuracy_score(y_test, y_pred)

    print("\n=== RuleClassifier Results ===")
    print(f"Accuracy = {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("[RuleClassifier] end")


if __name__ == "__main__":
    run_rules_baseline()