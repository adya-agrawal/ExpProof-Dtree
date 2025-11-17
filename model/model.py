# model_credit.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Model definition
# ---------------------------------------------------------
class Net(nn.Module):
    def __init__(self, d_in, d_hidden=16, d_out=8):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------
# Load dataset + train model
# ---------------------------------------------------------
def load_credit_and_train(epochs=800):

    print("Loading dataset...")

    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00350/default%20of%20credit%20card%20clients.xls"
    )
    df = pd.read_excel(url, header=1)

    y = df["default payment next month"].values.astype(np.int64)
    X_raw = df.drop(columns=["ID", "default payment next month"]).values.astype(np.float32)

    print("Dataset loaded:", X_raw.shape)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))

    model = Net(input_dim, d_hidden=16, d_out=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("Training model...")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.4f}")

    # test data kept as numpy arrays
    return model, X_train, y_train, X_test, y_test, input_dim