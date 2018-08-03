import torch
from torch import nn

class Logits(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, hidden, candidates: torch.Tensor):
    return torch.sum(torch.mul(torch.unsqueeze(hidden, 1),
                               candidates),
                     2)
