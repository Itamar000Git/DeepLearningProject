import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import re

from torch import optim

vocabulary_size = 0
word2location = {}
wordcounter = {}


def prepare_vocabulary(data, max_vocab=20000):
    global wordcounter, word2location, vocabulary_size


    for sentence in data:
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z\s]", " ", sentence)
        for word in sentence.split():
            wordcounter[word] = wordcounter.get(word, 0) + 1

    print("sorting")
    top_items = sorted(wordcounter.items(), key=lambda kv: kv[1], reverse=True)[:max_vocab]
    print("end sorting")

    top_words = [w for w, _ in top_items]

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
class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.linear3(x)


        out = torch.sigmoid(x)
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

    model = MLPModel(input_dim=20000)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    X_train_np = np.stack([convert2vec(r) for r in X_train]).astype(np.float32)
    data_x = torch.from_numpy(X_train_np)

    data_y = torch.tensor([1 if y == "positive" else 0 for y in y_train], dtype=torch.float32).unsqueeze(1)

    for epoch in range(10000):
        optimizer.zero_grad()  # Clear gradients w.r.t. parameters
        outputs = model(data_x)
        # print(outputs.shape)
        # print (data_y.shape)
        loss = criterion(outputs, data_y)  # Calculate the loss
        loss.backward()  # Getting gradients w.r.t. parameters
        optimizer.step()  # Updating parameters
        if epoch % 500 == 0:
            # Print out the loss
            print(f'Epoch [{epoch}/10000], Loss: {loss.item():.4f}')

    # x_test_vec = torch.tensor([convert2vec(r) for r in X_test],dtype=torch.float32)
    X_test_np = np.stack([convert2vec(r) for r in X_test]).astype(np.float32)
    x_test_vec = torch.from_numpy(X_test_np)

    y_test_vec = torch.tensor([1 if y == "positive" else 0 for y in y_test],dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        predictions = model(x_test_vec)

        # 2. המרה להחלטה בינארית (Thresholding)
        # כל מה שגדול מ-0.5 הופך ל-1, וכל השאר ל-0
        predicted_labels = (predictions > 0.5).float()

        # 3. המרה ל-Numpy (כי sklearn לא עובד ישירות עם טנסורים של PyTorch)
        y_true = y_test_vec.cpu().numpy()
        y_pred = predicted_labels.cpu().numpy()

        # 4. הדפסת אחוז דיוק (Accuracy)
        accuracy = accuracy_score(y_true, y_pred)
        print("\n=== FullyConnectedClassifier Results ===")
        print(f"Final Accuracy: {accuracy * 100:.2f}%")
        print("-" * 60)

        # 5. הדפסת טבלה מסודרת (Classification Report)
        # אפשר לשנות את השמות ב-target_names למה שמתאים (למשל: Negative, Positive)
        print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"], zero_division=0))


    print("[FullyConnectedClassifier] end")


if __name__ == "__main__":
    run_logisticModel()