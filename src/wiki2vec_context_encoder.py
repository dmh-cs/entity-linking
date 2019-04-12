from itertools import zip_longest

import torch
import torch.nn as nn
import numpy as np

class ContextEncoder(nn.Module):
  def __init__(self, wiki2vec, token_idx_lookup, device):
    super().__init__()
    self.wiki2vec = wiki2vec
    self.token_idx_lookup = token_idx_lookup
    self.dim_len = self.wiki2vec.dim_len
    self.lin = nn.Linear(self.dim_len, self.dim_len)
    self.device = device

  def _bag_to_tens(self, bag_of_nouns):
    longest = max(map(len, bag_of_nouns))
    token_idxs_by_bag = np.array([[self.token_idx_lookup[token]
                                   if (token is not None) and (token in self.token_idx_lookup)
                                   else 0
                                   for token, i in zip_longest(bag, range(longest))]
                                  for bag in bag_of_nouns])
    return self.wiki2vec(token_idxs_by_bag).to(self.device)


  def forward(self, bag_of_nouns):
    bag_tens = self._bag_to_tens(bag_of_nouns)
    sums = bag_tens.sum(1)
    normalized = sums / torch.norm(sums, dim=1).unsqueeze(1)
    return self.lin(normalized)
