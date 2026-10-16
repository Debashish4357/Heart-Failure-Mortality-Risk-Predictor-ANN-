
import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from ann_model import HeartFailureANN


# Create ANN
model = HeartFailureANN()
model.train()

# Loss function
loss_function = nn.BCELoss()

# Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# One example: 11 input features
X_sample = torch.randn(1, 11)

# Example actual target
y_sample = torch.tensor([[1.0]])

# 1. Clear old gradients
optimizer.zero_grad()

# 2. Forward pass
prediction = model(X_sample)

# 3. Calculate loss
loss = loss_function(prediction, y_sample)

# 4. Backpropagation
loss.backward()

# 5. Update weights
optimizer.step()

print("Prediction:", prediction.item())
print("Loss:", loss.item())
print("Backpropagation completed.")
print("Optimizer update completed.")
