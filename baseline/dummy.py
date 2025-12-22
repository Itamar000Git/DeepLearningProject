import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_dummy_baseline(
    path: str = r"Data\IMDB Dataset.csv",
    text_col: str = "review",
    label_col: str = "sentiment"
):
    print("[DummyBaseline] start")
    print(f"[DummyBaseline] reading: {path}")

    # 1) Load dataset
    df = pd.read_csv(path)  # Convert dataset to pandas DataFrame

    # print("Columns:", list(df.columns))
    # print("Rows:", len(df))
    # print(df.head(3))

    # 2) Choose columns + remove missing rows
    df = df.dropna(subset=[text_col, label_col]).copy()

    X_text = df[text_col].astype(str)  # reviews
    y = df[label_col].astype(str)      # labels: "positive"/"negative"

    # #label distribution
    # print("\nLabel value counts:")
    # print(y.value_counts().head(10))

    # 3) Train/Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) classification model: always predict most frequent class
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n=== DummyClassifier (most_frequent) Results ===")
    print(f"Accuracy = {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("[DummyBaseline] end")
