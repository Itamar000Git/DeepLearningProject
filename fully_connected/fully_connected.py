import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import re
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


# Global variables for text representation:
# vocabulary_size – number of words in the vocabulary
# word2location – maps each word to an index
# wordcounter – counts word frequency in training data
vocabulary_size = 0
word2location = {}
wordcounter = {}

# Builds a vocabulary from the training data only.
# Keeps only the most frequent words (up to max_vocab).
def prepare_vocabulary(data, max_vocab=20000):
    """
    Build Top-K vocabulary from TRAIN only.
    """
    global wordcounter, word2location, vocabulary_size

    # Reset dictionaries to avoid mixing vocabularies between runs.
    wordcounter = {}
    word2location = {}

    # Clean each sentence and count how many times each word appears.
    for sentence in data:
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z\s]", " ", sentence)
        for word in sentence.split():
            wordcounter[word] = wordcounter.get(word, 0) + 1

    # Sort words by frequency and keep the top max_vocab words.
    print("sorting")
    top_items = sorted(wordcounter.items(), key=lambda kv: kv[1], reverse=True)[:max_vocab]
    print("end sorting")

    # Assign a unique index to each word in the vocabulary.
    top_words = [w for w, _ in top_items]
    word2location = {w: i for i, w in enumerate(top_words)}
    vocabulary_size = len(word2location)
    return vocabulary_size

# Converts a review into a numerical vector using Bag-of-Words.
def convert2vec(sentence):
    # Converts a review into a numerical vector using Bag-of-Words.
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", " ", sentence)

    # Apply the same cleaning used during vocabulary creation.
    res_vec = np.zeros(vocabulary_size, dtype=np.float32)

    # Increase the count for each word that exists in the vocabulary.
    for word in sentence.split():
        idx = word2location.get(word)
        if idx is not None:
            res_vec[idx] += 1.0  # (counts). If you want presence only: res_vec[idx] = 1.0
    return res_vec



# Fully Connected Neural Network (MLP)
class MLPModel(nn.Module):
    # input_dim – size of the input vector (vocabulary size)
    # hidden1, hidden2 – sizes of hidden layers
    # dropout_p – dropout probability for regularization
    def __init__(self, input_dim, hidden1=128, hidden2=32, dropout_p=0.0):
        super().__init__()
        # Three fully connected layers.
        self.linear1 = nn.Linear(input_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout_p)

    # Forward pass of the network.
    def forward(self, x):

        # First hidden layer with ReLU (if negative keep in 0) and dropout.
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)

        # Second hidden layer with ReLU and dropout.
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)

        # Output layer returns logits (before sigmoid).
        logits = self.linear3(x)
        return logits


# Loads the dataset, trains the neural network,
# and evaluates it on the test set.
def run_fully_connected_nn(
    path: str = r"Data\IMDB Dataset.csv",
    text_col: str = "review",
    label_col: str = "sentiment",
    max_vocab: int = 20000,
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 2e-5,
    hidden1: int = 128,
    hidden2: int = 32,
    dropout_p: float = 0.2,
    seed: int = 42
):
    print("########### start fully connected model ###########")
    # Load the dataset and remove missing values.
    df = pd.read_csv(path).dropna(subset=[text_col, label_col]).copy()
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    # Split the data into training and test sets (80% / 20%),
    # while keeping the class distribution.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Build vocab on train only
    features = prepare_vocabulary(X_train, max_vocab=max_vocab)
    print("Vocabulary size:", features)

    # Vectorize
    # Convert all training reviews into numerical vectors
    X_train_np = np.stack([convert2vec(r) for r in X_train]).astype(np.float32)
    X_test_np  = np.stack([convert2vec(r) for r in X_test]).astype(np.float32)

    data_x = torch.from_numpy(X_train_np)
    test_x = torch.from_numpy(X_test_np)
    # Labels
    # Convert labels to binary format:
    # positive → 1, negative → 0
    train_y = torch.tensor([1 if yy == "positive" else 0 for yy in y_train],
                           dtype=torch.float32).unsqueeze(1)
    test_y = torch.tensor([1 if yy == "positive" else 0 for yy in y_test],
                          dtype=torch.float32).unsqueeze(1)

    # DataLoader (mini-batches)
    # mini-batch training improves optimization vs full-batch
    train_loader = DataLoader(
        TensorDataset(data_x, train_y),
        batch_size=batch_size,
        shuffle=True
    )

    # Model
    model = MLPModel(input_dim=features, hidden1=hidden1, hidden2=hidden2, dropout_p=dropout_p)

    # Loss + Optimizer
    # BCEWithLogitsLoss is numerically stable + works with logits
    criterion = nn.BCEWithLogitsLoss()

    # Adam converges faster/stabler than SGD for this setup
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    # Train the model for several epochs.
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)  # logits
            loss = criterion(logits, yb)  # targets shape (N,1)
            loss.backward() # Backpropagation and weight update.
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f}")  # FIX: proper loss print (no {})

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        probs = torch.sigmoid(logits)  # convert logits -> probabilities
        pred = (probs > 0.5).float()  # threshold -> 0/1

    # sklearn expects numpy arrays
    y_true = test_y.numpy()
    y_pred = pred.numpy()

    acc = accuracy_score(y_true, y_pred)
    print("\n=== FullyConnectedClassifier Results ===")
    print(f"Accuracy = {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("########### end fully connected model ###########")


if __name__ == "__main__":
    run_fully_connected_nn()