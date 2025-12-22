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


def prepare_vocabulary(data, max_vocab=20000):
    global wordcounter, word2location, vocabulary_size

    #wordcounter = {}
    #word2location = {}

    # 1) count words
    for sentence in data:
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z\s]", " ", sentence)
        for word in sentence.split():
            wordcounter[word] = wordcounter.get(word, 0) + 1

    # 2) take top-K words by frequency
    # sorted returns list of tuples: [(word, count), ...]
    print("sorting")
    top_items = sorted(wordcounter.items(), key=lambda kv: kv[1], reverse=True)[:max_vocab]
    print("end sorting")

    top_words = [w for w, _ in top_items]

    # 3) rebuild word2location with new compact indices
    word2location = {w: i for i, w in enumerate(top_words)}

    vocabulary_size = len(word2location)
    return vocabulary_size

def convert2vec(sentence):
    res_vec = np.zeros(vocabulary_size)
    for word in sentence.split(): #also here...
        if word in word2location:
            res_vec[word2location[word]] += 1
    return res_vec


# Define the model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


def run_logisticModel(
    path: str = r"Data\IMDB Dataset.csv",
    text_col: str = "review",
    label_col: str = "sentiment"
):
    df = pd.read_csv(path)
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    global vocabulary_size
    features = prepare_vocabulary(X_train)
    vocabulary_size =features

    #features = vocabulary_size
    # Initialize the model
    model = LogisticRegressionModel(features)

    # Loss and optimizer
    criterion = nn.BCELoss(reduction='mean')  # Binary Cross Entropy Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Stochastic Gradient Descent
    # Data preparation
    # Assuming convert2vec and data are defined somewhere above
    #
    # data_x = torch.tensor([convert2vec(r) for r in X_train],dtype=torch.float32)
    X_train_np = np.stack([convert2vec(r) for r in X_train]).astype(np.float32)
    data_x = torch.from_numpy(X_train_np)

    data_y = torch.tensor([1 if y == "positive" else 0 for y in y_train],dtype=torch.float32).unsqueeze(1)

    # Training the model
    model.train()

    for i in range(20000):
        # print(i)
        optimizer.zero_grad()  # Clear gradients w.r.t. parameters
        outputs = model(data_x)
        # print(outputs.shape)
        # print (data_y.shape)
        loss = criterion(outputs, data_y)  # Calculate the loss
        loss.backward()  # Getting gradients w.r.t. parameters
        optimizer.step()  # Updating parameters
        if i % 1000 == 0:
            # Print out the loss
            print(f'Loss:', {loss.item()})

    # x_test_vec = torch.tensor([convert2vec(r) for r in X_test],dtype=torch.float32)
    X_test_np = np.stack([convert2vec(r) for r in X_test]).astype(np.float32)
    x_test_vec = torch.from_numpy(X_test_np)

    y_test_vec = torch.tensor([1 if y == "positive" else 0 for y in y_test],dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        y_pred = model(x_test_vec)
        y_pred_bin = (y_pred > 0.5).float()



    acc = accuracy_score(y_test_vec, y_pred_bin)
    # acc = accuracy_score(y_test_vec.cpu().numpy(), y_pred_bin.cpu().numpy())

    print("\n=== logisticModelClassifier Results ===")
    print(f"Accuracy = {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test_vec, y_pred_bin, zero_division=0))

    print("[RuleClassifier] end")


if __name__ == "__main__":
    run_logisticModel()