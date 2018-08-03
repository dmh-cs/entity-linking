import torch
from torch import nn
from torch.nn import functional as F

import math


class AdaptiveLogits(nn.Module):
  """Logits calculation
  Args:
    candidates: tensor containing the vector representation of the candidates (eg word embeddings) sorted by frequency
    cutoffs: list of cutoff indices for each cluster when words are sorted by decreasing frequency
    reduce_factor: dimension reduction factor of each tail bucket. Default: 4

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
    >>> hidden = torch.randn(batch_size, hidden_size)
    >>> targets = torch.randint(low=0, high=vocab_size + 1, size=[128])
    >>> all_logits = adaptive_logits(hidden, targets)
    >>> loss = m.loss(all_logits, targets)
  """

  def __init__(self, candidates, cutoffs, reduce_factor=4):
    super().__init__()
    self.id = []
    self.cutoffs = cutoffs
    self.head = self._get_head_calc(candidates, cutoffs)
    self.tail = self._get_tail_calc(candidates, cutoffs, reduce_factor)

  def _get_head_calc(self, candidates, cutoffs):
    all_logits_size = cutoffs[0] + len(cutoffs) - 1
    hidden_size = candidates.shape[1]
    head_calc = nn.Linear(hidden_size, all_logits_size, bias=False)
    tail_vectors = torch.Tensor(len(cutoffs[1:]), hidden_size)
    tail_vectors.normal_(0, 1.0/math.sqrt(hidden_size))
    weights = torch.concatenate((candidates[:cutoffs[0]], tail_vectors), 0)
    head_calc.weight = torch.Parameter(torch.transpose(weights, 0, 1))
    return head_calc

  def _get_tail_calc(self, candidates, cutoffs, reduce_factor):
    hidden_size = candidates.shape[1]
    tail = nn.ModuleList()
    for i in range(len(cutoffs) - 1):
      if reduce_factor == 1:
        seq = nn.Linear(hidden_size, cutoffs[i + 1] - cutoffs[i], bias=False)
        seq.weight = torch.transpose(candidates[cutoffs[i] : cutoffs[i + 1]], 0, 1)
      else:
        down = nn.Linear(hidden_size,
                         hidden_size // reduce_factor ** i,
                         bias=False)
        decode = nn.Linear(hidden_size // reduce_factor ** i,
                           cutoffs[i + 1] - cutoffs[i],
                           bias=False)
        decode.weight = torch.transpose(candidates[cutoffs[i] : cutoffs[i + 1]], 0, 1)
        seq = nn.Sequential(down, decode)
      tail.append(seq)
    return tail

  def _set_targets(self, targets):
    self.id = []
    for i in range(len(self.cutoffs) - 1):
      mask = targets.ge(self.cutoffs[i]).mul(targets.lt(self.cutoffs[i + 1]))
      if mask.any():
        self.id.append(mask.float().nonzero().squeeze(1))
      else:
        self.id.append(None)

  def forward(self, hidden, targets):
    all_logits = [self.head(hidden)]
    self._set_targets(targets)
    for i in range(len(self.id)):
      if self.id[i] is not None:
        all_logits.append(self.tail[i](hidden.index_select(0, self.id[i])))
      else:
        all_logits.append(None)
    return all_logits

  def _remap_targets(self, targets):
    new_targets = [targets.clone()]
    for i in range(len(self.cutoffs) - 1):
      mask = targets.ge(self.cutoffs[i]).mul(targets.lt(self.cutoffs[i + 1]))
      new_targets[0][mask] = self.cutoffs[0] + i
      if mask.any():
        new_targets.append(targets[mask].add(-self.cutoffs[i]))
      else:
        new_targets.append(None)
    return new_targets

  def loss(self, all_logits, targets):
    batch_size = all_logits[0].size(0)
    targets = self._remap_targets(targets.data)
    output = 0.0
    for i in range(len(all_logits)):
      if all_logits[i] is not None:
        assert targets[i].min() >= 0 and targets[i].max() <= all_logits[i].size(1)
        output = output + F.cross_entropy(all_logits[i],
                                          targets[i],
                                          size_average=False)
    output /= batch_size
    return output
