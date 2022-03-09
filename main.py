import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from models.gcn import GCN
from data.utils import load_data

from metrics import masked_accuracy, masked_softmax_cross_entropy

# 1. argparse
parser = argparse.ArgumentParser(description='[GNN] Graph Neural Network Implementations')
parser.add_argument("--dataset", type=str, required=True, default="cora", help=f"Dataset name")
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. prepare data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)

features = torch.Tensor(features.toarray())
y_train, y_val, y_test = torch.Tensor(y_train), torch.Tensor(y_val), torch.Tensor(y_test)

NUM_NODES = adj.shape[0]
IN_FEATURES = features.shape[1]
OUT_FEATURES = y_train.shape[1]

# 3. prepare model
model = GCN(adj, in_features=IN_FEATURES, out_features=OUT_FEATURES)
model.to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-02, weight_decay=5e-4)
loss_fn = masked_softmax_cross_entropy

# 4. train
EPOCH = 100
for e in range(EPOCH):
    model.train()
    X, y = features, y_train
    y_hat = model(X)

    optimizer.zero_grad()
    train_loss = loss_fn(y_hat, y, train_mask)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        X, y = features, y_val
        y_hat = model(X)
        val_loss = loss_fn(y_hat, y, val_mask)

        prediction = torch.argmax(y_hat, dim=1)
        y_label = torch.argmax(y, dim=1)
        accuracy = masked_accuracy(prediction, y_label, val_mask)

    print(f"epoch {e} | train_loss {train_loss}, val_loss {val_loss}, val_accuracy {accuracy}")

# 5. test
model.eval()
with torch.no_grad():
    X, y = features, y_test
    y_hat = model(X)
    test_loss = loss_fn(y_hat, y, test_mask)

    prediction = torch.argmax(y_hat, dim=1)
    y_label = torch.argmax(y, dim=1)
    accuracy = masked_accuracy(prediction, y_label, test_mask)

    print(f"Test: accuracy {accuracy}, CE {test_loss}")