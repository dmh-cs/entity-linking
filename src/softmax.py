from torch import nn

class Softmax(nn.Module):
  def __init__(self):
    super().__init__()
    self.softmax = nn.functional.log_softmax

  def forward(self, logits):
    return self.softmax(logits, dim=1)
