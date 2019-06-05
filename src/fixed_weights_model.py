import torch
from torch import nn

class FixedWeights(nn.Module):
  def __init__(self, weights):
    super().__init__()
    self.weights = torch.tensor(weights, dtype=torch.float32, requires_grad=False)

  def forward(self, x):
    return (self.weights.unsqueeze(0) * x).sum(1)
