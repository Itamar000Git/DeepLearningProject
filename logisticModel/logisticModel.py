import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import re

vocabulary_size = 0
word2location = {}
wordcounter = {}



def clean_and_tokenize(text):
    """
    פונקציה שמנקה את הטקסט ומחלקת למילים.
    חשוב להשתמש באותה פונקציה גם באימון וגם בטסט כדי למנוע אי-התאמות.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def get_ngrams(tokens, n):
    """
    פונקציה שמקבלת רשימת מילים ומחזירה רשימה של n-grams.
    למשל עבור ['very', 'good', 'movie'] ו-n=2 נקבל ['very good', 'good movie']
    """
    if n < 2:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def prepare_vocabulary(data, max_vocab=20000, n_gram=2):
    global wordcounter, word2location, vocabulary_size

    wordcounter = {}  # איפוס המונה
    word2location = {}

    print("Counting words and n-grams...")
    for sentence in data:
        tokens = clean_and_tokenize(sentence)

        # 1. ספירת מילים בודדות
        for word in tokens:
            wordcounter[word] = wordcounter.get(word, 0) + 1

        # 2. ספירת צירופים (N-grams)
        if n_gram > 1:
            ngrams_list = get_ngrams(tokens, n_gram)
            for bg in ngrams_list:
                wordcounter[bg] = wordcounter.get(bg, 0) + 1

    # מיון ולקחת ה-Top K השכיחים ביותר
    print("Sorting vocabulary...")
    top_items = sorted(wordcounter.items(), key=lambda kv: kv[1], reverse=True)[:max_vocab]

    # בניית המילון (מילה -> אינדקס)
    top_words = [w for w, _ in top_items]
    word2location = {w: i for i, w in enumerate(top_words)}

    vocabulary_size = len(word2location)
    print(f"Vocabulary size created: {vocabulary_size}")
    return vocabulary_size


def convert2vec(sentence, n_gram=2):
    """
    ממיר משפט לווקטור על סמך המילון שנבנה.
    """
    res_vec = np.zeros(vocabulary_size)
    tokens = clean_and_tokenize(sentence)

    # בדיקת מילים בודדות
    for word in tokens:
        if word in word2location:
            res_vec[word2location[word]] += 1

    # בדיקת N-grams
    if n_gram > 1:
        ngrams_list = get_ngrams(tokens, n_gram)
        for bg in ngrams_list:
            if bg in word2location:
                res_vec[word2location[bg]] += 1

    return res_vec


# --- המודל ---

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


# --- הפונקציה הראשית ---

def run_logisticModel(
        path: str = r"Data\IMDB Dataset.csv",
        text_col: str = "review",
        label_col: str = "sentiment"
):
    print("Loading data...")
    # נסה לטעון, ואם הנתיב לא קיים ניצור דאטה דמי לצורך הדגמה
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: File not found at {path}. Using dummy data.")
        df = pd.DataFrame({
            "review": ["good movie", "bad movie", "very good", "not good at all", "excellent film"] * 20,
            "sentiment": ["positive", "negative", "positive", "negative", "positive"] * 20
        })

    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    # חלוקה לטסט וטריין
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    global vocabulary_size

    # כאן אנחנו קובעים שנשתמש גם ב-Bigrams (n_gram=2)
    current_n_gram = 2
    features = prepare_vocabulary(X_train, max_vocab=20000, n_gram=current_n_gram)
    vocabulary_size = features

    # אתחול המודל
    model = LogisticRegressionModel(features)

    # Loss and optimizer
    criterion = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # הכנת הדאטה לאימון
    print("Vectorizing train data (this might take a moment)...")
    # המרה ל-numpy ואז ל-tensor כדי לחסוך זיכרון וזמן
    X_train_np = np.stack([convert2vec(r, n_gram=current_n_gram) for r in X_train]).astype(np.float32)
    data_x = torch.from_numpy(X_train_np)

    data_y = torch.tensor([1 if label == "positive" else 0 for label in y_train], dtype=torch.float32).unsqueeze(1)

    # Training the model
    print("Starting training...")
    model.train()


    epochs = 30000

    for i in range(epochs):
        optimizer.zero_grad()
        outputs = model(data_x)
        loss = criterion(outputs, data_y)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f'Epoch {i}, Loss: {loss.item():.4f}')

    # הכנת הדאטה לטסט
    print("Vectorizing test data...")
    X_test_np = np.stack([convert2vec(r, n_gram=current_n_gram) for r in X_test]).astype(np.float32)
    x_test_vec = torch.from_numpy(X_test_np)

    y_test_vec = torch.tensor([1 if label == "positive" else 0 for label in y_test], dtype=torch.float32).unsqueeze(1)

    # ביצוע בדיקה
    model.eval()  # מעבר למצב הערכה
    with torch.no_grad():
        y_pred = model(x_test_vec)
        y_pred_bin = (y_pred > 0.5).float()

    acc = accuracy_score(y_test_vec, y_pred_bin)

    print("\n=== logisticModelClassifier Results ===")
    print(f"Accuracy = {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test_vec, y_pred_bin, zero_division=0))

    print("[RuleClassifier] end")


if __name__ == "__main__":
    run_logisticModel()