import os
import time
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Rule 6: fixed seed
# =========================
SEED = 25
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# Paths
# =========================
BASE = "data/public"
RUN_NAME = "run1_gcn_baseline"

# =========================
# Timing
# =========================
start_time = time.time()

# =========================
# Load data
# =========================
X_np = np.load(f"{BASE}/node_features.npy")
A_sp = sp.load_npz(f"{BASE}/adjacency_matrix.npz").tocsr()
train_df = pd.read_csv(f"{BASE}/train_target.csv")
test_df = pd.read_csv(f"{BASE}/test_target_without_labels.csv")

num_nodes, num_features = X_np.shape
num_classes = train_df["ml_target"].nunique()

print("X shape:", X_np.shape)
print("A shape:", A_sp.shape)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# =========================
# Train / validation split
# =========================
all_train_ids = train_df["id"].to_numpy()
all_train_labels = train_df["ml_target"].to_numpy()

perm = np.random.permutation(len(all_train_ids))
split = int(0.9 * len(all_train_ids))

train_idx_np = all_train_ids[perm[:split]]
val_idx_np = all_train_ids[perm[split:]]

y_all = np.full(num_nodes, -1, dtype=np.int64)
y_all[all_train_ids] = all_train_labels

# =========================
# Normalize adjacency
# A_hat = D^{-1/2}(A+I)D^{-1/2}
# =========================
A_sp = A_sp + sp.eye(num_nodes, format="csr")
deg = np.array(A_sp.sum(axis=1)).flatten()
deg_inv_sqrt = np.power(deg, -0.5)
deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
D_inv_sqrt = sp.diags(deg_inv_sqrt)

A_norm = D_inv_sqrt @ A_sp @ D_inv_sqrt
A_norm = A_norm.tocoo()

indices = torch.tensor(
    np.vstack((A_norm.row, A_norm.col)),
    dtype=torch.long
)
values = torch.tensor(A_norm.data, dtype=torch.float32)
A_torch = torch.sparse_coo_tensor(
    indices, values, (num_nodes, num_nodes)
).coalesce()

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_all, dtype=torch.long)

train_idx = torch.tensor(train_idx_np, dtype=torch.long)
val_idx = torch.tensor(val_idx_np, dtype=torch.long)
test_idx = torch.tensor(test_df["id"].to_numpy(), dtype=torch.long)

# =========================
# GCN model
# =========================
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.sparse.mm(adj, x)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adj, x)
        x = self.lin2(x)
        return x

model = GCN(
    in_dim=num_features,
    hidden_dim=64,
    out_dim=num_classes,
    dropout=0.5
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

class_counts = train_df["ml_target"].value_counts().sort_index().to_numpy()
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# =========================
# Train with early stopping
# =========================
epochs = 100
patience = 15
best_val_loss = float("inf")
best_state = None
patience_counter = 0

train_losses = []
val_losses = []
val_accs = []

for epoch in range(1, epochs + 1):
    model.train()
    logits = model(X, A_torch)
    train_loss = criterion(logits[train_idx], y[train_idx])

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_eval = model(X, A_torch)
        val_loss = criterion(logits_eval[val_idx], y[val_idx]).item()
        val_pred = logits_eval[val_idx].argmax(dim=1)
        val_acc = (val_pred == y[val_idx]).float().mean().item()

    train_losses.append(float(train_loss.item()))
    val_losses.append(float(val_loss))
    val_accs.append(float(val_acc))

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss.item():.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# =========================
# Restore best model
# =========================
if best_state is not None:
    model.load_state_dict(best_state)

# =========================
# Predict test
# =========================
model.eval()
with torch.no_grad():
    final_logits = model(X, A_torch)
    test_pred = final_logits[test_idx].argmax(dim=1).cpu().numpy()

submission = pd.DataFrame({
    "id": test_df["id"],
    "ml_target": test_pred.astype(int)
})
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")

# =========================
# Save loss history
# =========================
loss_df = pd.DataFrame({
    "epoch": np.arange(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_acc": val_accs,
})
loss_df.to_csv("loss_history.csv", index=False)
print("Saved loss_history.csv")
print("loss_history.csv saved. Plot can be generated later if matplotlib is fixed.")

# =========================
# Save run summary
# =========================
elapsed = time.time() - start_time

with open("run_summary.txt", "w") as f:
    f.write(f"Run name: {RUN_NAME}\n")
    f.write(f"Seed: {SEED}\n")
    f.write("Model: GCN\n")
    f.write("Hidden dim: 64\n")
    f.write("Learning rate: 0.01\n")
    f.write("Dropout: 0.5\n")
    f.write(f"Epochs completed: {len(train_losses)}\n")
    f.write(f"Best val loss: {best_val_loss:.6f}\n")
    f.write(f"Last val acc: {val_accs[-1]:.6f}\n")
    f.write(f"Training time (seconds): {elapsed:.2f}\n")

print("Saved run_summary.txt")
print(f"Training time: {elapsed:.2f} seconds")
print(submission.head())
