import torch
import torch.nn as nn

from weighted_sum_encoder import WeightedSumEncoder
from bow_helpers import collate_bow_doc

class ContextEncoderModel(nn.Module):
  def __init__(self, word_embeds, use_cnts=False, idf=None):
    super().__init__()
    self.context_encoder = WeightedSumEncoder(word_embeds, use_cnts=use_cnts, idf=idf)
    self.device = torch.device('cpu')

  def to(self, *args, **kwargs):
    device = args[0]
    super().to(*args, **kwargs)
    self.device = device

  def forward(self, context, doc):
    return self.context_encoder(collate_bow_doc(context, device=self.device))
