import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from adaptive_logits import AdaptiveLogits
from adaptive_softmax import AdaptiveSoftmax

class Toy(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Linear(1, 1)
    self.relu1 = nn.ReLU()
    self.second = nn.Linear(1, 1)
    self.relu2 = nn.ReLU()
    self.third = nn.Linear(1, 1)
    self.relu3 = nn.ReLU()

  def forward(self, data):
    x = torch.unsqueeze(torch.arange(len(data[0]), dtype=torch.float, device=data[1][0].device), 1)
    return self.relu3(self.third(self.relu2(self.second(self.relu1(self.first(x))))))

def train():
  cutoffs = [10, 20, 300 + 1]
  embeds = nn.Embedding(300, 100)
  adaptive_logits = AdaptiveLogits(embeds, cutoffs)
  adaptive_softmax = AdaptiveSoftmax(adaptive_logits)
  model = Toy(embeds)
  x = torch.randn((3, 100))
  targets = torch.tensor([0, 1, 10])
  optimizer = optim.Adam(itertools.chain(model.parameters(),
                                         # adaptive_logits.parameters()))
                                         embeds.parameters()))
  for i in range(1000):
    optimizer.zero_grad()
    hidden = model(x)
    logits = torch.mm(hidden, torch.transpose(embeds(torch.tensor(range(300))), 0, 1))
    loss = F.cross_entropy(logits, targets)
    # logits = adaptive_logits(hidden, targets)
    # loss = adaptive_logits.loss(logits, targets)
    loss.backward()
    optimizer.step()
  # print(torch.argmax(adaptive_softmax(hidden), 1))
  print(torch.argmax(F.softmax(logits, 1), 1))


def main():
  train()

if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
