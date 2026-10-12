
import torch
import torch.nn as nn

# Binary Cross-Entropy loss
loss_function = nn.BCELoss()

# Example 1: Actual outcome is 1
actual_1 = torch.tensor([[1.0]])
prediction_1 = torch.tensor([[0.90]])

loss_1 = loss_function(prediction_1, actual_1)

# Example 2: Actual outcome is 1, but prediction is poor
actual_2 = torch.tensor([[1.0]])
prediction_2 = torch.tensor([[0.20]])

loss_2 = loss_function(prediction_2, actual_2)

print("Loss for good prediction:", loss_1.item())
print("Loss for poor prediction:", loss_2.item())
