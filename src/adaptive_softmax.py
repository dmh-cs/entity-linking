import torch
from torch import nn
from torch.nn.functional import softmax

from adaptive_logits import AdaptiveLogits

import math


class AdaptiveSoftmax(object):
  """Adaptive Softmax calculation
  This is NOT a drop-in replacement for nn.functional.softmax.

  Args:
    adaptive_logits: instance of `AdaptiveLogits`

  Shape:
    - hidden: (batch_size, hidden_size)
    - targets: (batch_size)
    - candidates: (num_candidates, hidden_size)
    - all_logits: [(batch_size, cutoffs[0] + len(cutoffs) - 1), (batch_size * p_t1, cutoffs[1] - cutoffs[0]), ...]

  Attributes:
    head: the learnable weights of the module for head bucket
    tail: the learnable weights of the module for tail buckets

  Example:

    >>> candidates = nn.Embedding(num_candidates, hidden_size)
    >>> cutoffs = [2000, 10000, vocab_size + 1];
    >>> adaptive_logits = AdaptiveLogits(candidates, cutoffs)
    >>> adaptive_softmax = AdaptiveSoftmax(adaptive_logits)
    >>> hidden = torch.randn(batch_size, hidden_size)
    >>> probs = adaptive_softmax(hidden)
  """

  def __init__(self, adaptive_logits):
    super().__init__()
    self.adaptive_logits = adaptive_logits
    self.head = self.adaptive_logits.head
    self.tail = self.adaptive_logits.tail
    self.cutoffs = self.adaptive_logits.cutoffs

  def forward(self, hidden):
    with torch.no_grad():
      head_out = self.head(hidden)
      batch_size = head_out.size(0)
      prob = torch.empty(batch_size, self.cutoffs[-1], device=hidden.device)
      lsm_head = softmax(head_out, 1)
      prob[:, : self.cutoffs[0]].copy_(lsm_head[:, : self.cutoffs[0]])
      for i in range(len(self.tail)):
        split = lsm_head[:, self.cutoffs[0] + i].unsqueeze(1)
        lsm_tail = softmax(self.tail[i](hidden), 1)
        prob[:, self.cutoffs[i] : self.cutoffs[i + 1]].copy_(lsm_tail).add_(split)
    return prob
