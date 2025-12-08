import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


CSV_FILE = "f1_2025_processed.csv"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42




random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)




df = pd.read_csv(CSV_FILE)

# Columns that are NOT used as numeric features
meta_cols = [
    "year",
    "round",
    "race_slug",
    "race_name",
    "driver_code",
    "driver_name",
    "team_name",
]

label_col = "is_winner"

# Everything else is treated as a feature
feature_cols = [c for c in df.columns if c not in meta_cols + [label_col]]

print("Using feature columns:")
for c in feature_cols:
    print("  -", c)

# Split by race round (prevents leakage)
train_df = df[df["round"] <= 10].reset_index(drop=True)
val_df   = df[(df["round"] >= 11) & (df["round"] <= 12)].reset_index(drop=True)
test_df  = df[df["round"] >= 13].reset_index(drop=True)

print(f"\nTrain rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
y_train = train_df[label_col].to_numpy(dtype=np.float32)

X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
y_val = val_df[label_col].to_numpy(dtype=np.float32)

X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
y_test = test_df[label_col].to_numpy(dtype=np.float32)




mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)
std[std == 0.0] = 1.0  # avoid division by zero

X_train_norm = (X_train - mean) / std
X_val_norm   = (X_val - mean) / std
X_test_norm  = (X_test - mean) / std

X_train_t = torch.from_numpy(X_train_norm)
y_train_t = torch.from_numpy(y_train).unsqueeze(1)  # (N, 1)

X_val_t = torch.from_numpy(X_val_norm)
y_val_t = torch.from_numpy(y_val).unsqueeze(1)

X_test_t = torch.from_numpy(X_test_norm)
y_test_t = torch.from_numpy(y_test).unsqueeze(1)




train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# The actuable baseline model 

class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # output logits
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train_t.shape[1]
model = BaselineMLP(input_dim).to(DEVICE)

# Handle class imbalance: pos_weight = (#neg / #pos)
num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
pos_weight_value = float(num_neg / max(num_pos, 1.0))
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nUsing pos_weight={pos_weight_value:.3f} for BCEWithLogitsLoss")
print(f"Training on device: {DEVICE}")




def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)

            all_logits.append(logits)
            all_targets.append(yb)

    
    all_logits = torch.cat(all_logits, dim=0)        # shape (N, 1)
    all_targets = torch.cat(all_targets, dim=0)      # shape (N, 1)

    # Compute loss
    loss = criterion(all_logits, all_targets).item()

    
    probs = torch.sigmoid(all_logits).cpu()
    preds = (probs >= 0.5).float()
    targets_cpu = all_targets.cpu()

    acc = accuracy_score(targets_cpu.numpy(), preds.numpy())

    return loss, acc



# The trainingg loop

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
    val_loss, val_acc = evaluate(model, val_loader)

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )


# Final evaluation on test set

test_loss, test_acc = evaluate(model, test_loader)
print("\n Final Test Performance ")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
