import torch
import torch.nn as nn

from weighted_sum_encoder import WeightedSumEncoder
from mlp import MLP
from bow_helpers import collate_bow_doc

class MentionEncoderModel(nn.Module):
  def __init__(self, word_embeds, dropout_keep_prob, use_cnts=False):
    super().__init__()
    word_embed_len = word_embeds.weight.shape[1]
    self.context_encoder = WeightedSumEncoder(word_embeds, use_cnts=use_cnts)
    self.doc_encoder = WeightedSumEncoder(word_embeds, use_cnts=use_cnts)
    self.mlp = MLP(2 * word_embed_len, word_embed_len, [], dropout_keep_prob)
    self.device = torch.device('cpu')

  def to(self, *args, **kwargs):
    device = args[0]
    super().to(*args, **kwargs)
    self.device = device

  def forward(self, context, doc):
    return self.mlp(torch.cat([self.context_encoder(collate_bow_doc(context,
                                                                    device=self.device)),
                               self.doc_encoder(collate_bow_doc(doc,
                                                                device=self.device))],
                              1))
