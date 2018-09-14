import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveLogits(nn.Module):
  """Logits calculation
  Args:
    vocab: tensor containing the vector representation of the vocabulary (eg the word embeddings) sorted by frequency
    cutoffs: list of cutoff indices for each cluster when words are sorted by decreasing frequency
    reduce_factor: dimension reduction factor of each tail bucket. Default: 4

  Shape:
    - hidden: (batch_size, hidden_size)
    - targets: (batch_size)
    - vocab: (vocab_size, hidden_size)
    - all_logits: [(batch_size, cutoffs[0] + len(cutoffs) - 1), (batch_size * p_t1, cutoffs[1] - cutoffs[0]), ...]

  Attributes:
    head: the learnable weights of the module for head bucket
    tail: the learnable weights of the module for tail buckets
  """

  def __init__(self, embeds, cutoffs, reduce_factor=4, device=None):
    super().__init__()
    if device is None:
      self.device = embeds.weight.device
    else:
      self.device = device
    self.other_modules = nn.ModuleList()
    self.other_params = nn.ParameterList()
    self.id = []
    self.cutoffs = cutoffs
    self.embeds = embeds
    self.order = torch.tensor(range(len(embeds.weight)), device=self.device)
    self.head = self._get_head_calc(cutoffs)
    self.tail = self._get_tail_calc(cutoffs, reduce_factor)

  def _get_head_calc(self, cutoffs):
    hidden_size = self.embeds.weight.shape[1]
    tail_vectors = nn.Linear(hidden_size, len(cutoffs[1:])).to(self.device)
    shortlist_bias = nn.Parameter(torch.randn(cutoffs[0]))
    self.other_modules.append(tail_vectors)
    self.other_params.append(shortlist_bias)
    def head_calc(hidden):
      shortlist = self.embeds(self.order[:cutoffs[0]])
      shortlist_result = torch.mm(hidden, torch.transpose(shortlist, 0, 1)) + torch.unsqueeze(shortlist_bias, 0)
      tail_vectors_result = tail_vectors(hidden)
      return torch.cat((shortlist_result, tail_vectors_result), 1)
    return head_calc

  def _get_tail_calc(self, cutoffs, reduce_factor):
    hidden_size = self.embeds.weight.shape[1]
    tail = []
    for i in range(len(cutoffs) - 1):
      tail_bias = nn.Parameter(torch.randn(cutoffs[i + 1] - cutoffs[i]))
      self.other_params.append(tail_bias)
      if reduce_factor == 1:
        def seq(hidden, tail_bias=tail_bias, i=i):
          tail_cluster = self.embeds(self.order[cutoffs[i] : cutoffs[i + 1]])
          return torch.mm(hidden, torch.transpose(tail_cluster, 0, 1)) + torch.unsqueeze(tail_bias, 0)
      else:
        down = nn.Linear(hidden_size,
                         hidden_size // reduce_factor ** i,
                         bias=False).to(self.device)
        self.other_modules.append(down)
        def seq(hidden, tail_bias=tail_bias, down=down, i=i):
          decode_weight = down(self.embeds(self.order[cutoffs[i] : cutoffs[i + 1]]))
          return torch.mm(down(hidden), torch.transpose(decode_weight, 0, 1)) + torch.unsqueeze(tail_bias, 0)
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
        assert targets[i].min() >= 0 and targets[i].max() < all_logits[i].size(1)
        output = output + F.cross_entropy(all_logits[i],
                                          targets[i],
                                          size_average=False)
    output /= batch_size
    return output
