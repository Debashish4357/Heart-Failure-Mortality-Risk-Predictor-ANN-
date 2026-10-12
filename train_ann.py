
import copy
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ann_model import HeartFailureANN


# ============================================================
# STEP 8.4 — ANN TRAINING LOOP
# ============================================================

# Reproducibility
torch.manual_seed(42)

# Load dataset
df = pd.read_csv(
    "data/heart_failure_clinical_records_dataset.csv"
)

# Separate features and target
# Exclude leakage-prone 'time' column
X = df.drop(columns=["DEATH_EVENT", "time"])
y = df["DEATH_EVENT"]


# Use the first CV fold for this training exercise
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_index, val_index = next(skf.split(X, y))

X_train = X.iloc[train_index]
X_val = X.iloc[val_index]

y_train = y.iloc[train_index]
y_val = y.iloc[val_index]


# Scale using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)

y_val_tensor = torch.tensor(
    y_val.to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)


# Create batches
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# Create ANN
model = HeartFailureANN()

# Loss and optimizer
loss_function = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)


# Training settings
epochs = 100

print("=" * 60)
print("ANN TRAINING")
print("=" * 60)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Input features:", X_train.shape[1])


# Training loop
for epoch in range(epochs):

    model.train()
    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_X)

        # Calculate loss
        loss = loss_function(predictions, batch_y)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_train_loss += loss.item() * len(batch_X)

    average_train_loss = (
        total_train_loss / len(train_dataset)
    )


    # Validation: no weight updates
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val_tensor)

        val_loss = loss_function(
            val_predictions,
            y_val_tensor
        ).item()


    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )


print("\nANN training completed.")





