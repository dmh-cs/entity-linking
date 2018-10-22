import torch

class Logits():
  def __call__(self, hidden, candidates: torch.Tensor):
    return torch.sum(torch.mul(torch.unsqueeze(hidden, 1),
                               candidates),
                     2)
