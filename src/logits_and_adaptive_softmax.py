import torch
from torch import nn
from torch.nn import functional as F

import math


class LogitsAndAdaptiveSoftmax(nn.Module):
  """Logits calculation and Adaptive Softmax output layer
  This is NOT a drop-in replacement for nn.functional.softmax. See the usage of `LogitsAndSoftmax`

  Args:
    candidates: tensor containing the vector representation of the candidates (eg word embeddings) sorted by frequency
    cutoffs: list of cutoff indices for each cluster when words are sorted by decreasing frequency
    reduce_factor: dimension reduction factor of each tail bucket. Default: 4

  Shape:
    - hidden: (batch_size, hidden_size)
    - target: (batch_size)
    - candidates: (num_candidates, hidden_size)
    - all_logits: [(batch_size, cutoffs[0] + len(cutoffs) - 1), (batch_size * p_t1, cutoffs[1] - cutoffs[0]), ...]

  Attributes:
    head: the learnable weights of the module for head bucket
    tail: the learnable weights of the module for tail buckets

  Example:

    >>> cutoffs = [2000, 10000, vocab_size + 1];
    >>> m = AdaptiveSoftmax(hidden_size, cutoffs)
    >>> hidden = torch.randn(batch_size, hidden_size)
    >>> target = torch.randint(low=0, high=vocab_size + 1, size=[128])
    >>> all_logits = m(hidden, target)
    >>> log_prob = m.log_prob(hidden)
    >>> loss = m.loss(all_logits, target)
  """

  def __init__(self, candidates, cutoffs, reduce_factor=4):
    super().__init__()
    self.id = []
    self.cutoffs = cutoffs
    all_logits_size = cutoffs[0] + len(cutoffs) - 1
    hidden_size = candidates.shape[1]
    self.head = nn.Linear(hidden_size, all_logits_size, bias=False)
    tail_vectors = torch.Tensor(len(cutoffs[1:]), hidden_size)
    tail_vectors.normal_(0, 1.0/math.sqrt(hidden_size))
    weights = torch.concatenate((candidates[:cutoffs[0]], tail_vectors), 0)
    self.head.weight = torch.Parameter(torch.transpose(weights, 0, 1))
    self.tail = nn.ModuleList()
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
      self.tail.append(seq)

  def _set_target(self, target):
    self.id = []
    for i in range(len(self.cutoffs) - 1):
      mask = target.ge(self.cutoffs[i]).mul(target.lt(self.cutoffs[i + 1]))
      if mask.any():
        self.id.append(mask.float().nonzero().squeeze(1))
      else:
        self.id.append(None)

  def forward(self, hidden, target):
    all_logits = [self.head(hidden)]
    self._set_target(target)
    for i in range(len(self.id)):
      if self.id[i] is not None:
        all_logits.append(self.tail[i](hidden.index_select(0, self.id[i])))
      else:
        all_logits.append(None)
    return all_logits

  def log_prob(self, hidden):
    with torch.no_grad():
      head_out = self.head(hidden)
      batch_size = head_out.size(0)
      prob = torch.empty(batch_size, self.cutoffs[-1], device=hidden.device)
      lsm_head = F.log_softmax(head_out, 1)
      prob[:, : self.cutoffs[0]].copy_(lsm_head[:, : self.cutoffs[0]])
      for i in range(len(self.tail)):
        split = lsm_head[:, self.cutoffs[0] + i].unsqueeze(1)
        lsm_tail = F.log_softmax(self.tail[i](hidden), 1)
        prob[:, self.cutoffs[i] : self.cutoffs[i + 1]].copy_(lsm_tail).add_(split)
    return prob

  def _remap_target(self, target):
    new_target = [target.clone()]
    for i in range(len(self.cutoffs) - 1):
      mask = target.ge(self.cutoffs[i]).mul(target.lt(self.cutoffs[i + 1]))
      new_target[0][mask] = self.cutoffs[0] + i
      if mask.any():
        new_target.append(target[mask].add(-self.cutoffs[i]))
      else:
        new_target.append(None)
    return new_target

  def loss(self, all_logits, target):
    batch_size = all_logits[0].size(0)
    target = self._remap_target(target.data)
    output = 0.0
    for i in range(len(all_logits)):
      if all_logits[i] is not None:
        assert target[i].min() >= 0 and target[i].max() <= all_logits[i].size(1)
        output = output + F.cross_entropy(all_logits[i],
                                          target[i],
                                          size_average=False)
    output /= batch_size
    return output
