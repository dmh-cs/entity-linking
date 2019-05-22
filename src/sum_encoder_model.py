import torch
import torch.nn as nn

from weighted_sum_encoder import WeightedSumEncoder
from mlp import MLP

class MentionEncoderModel(nn.Module):
  def __init__(self, word_embeds, dropout_keep_prob):
    super().__init__()
    word_embed_len = word_embeds.weight.shape[1]
    self.context_encoder = WeightedSumEncoder(word_embeds)
    self.doc_encoder = WeightedSumEncoder(word_embeds)
    self.mlp = MLP(2 * word_embed_len, word_embed_len, [], dropout_keep_prob)

  def forward(self, context, doc):
    return self.mlp(torch.cat([self.context_encoder(context),
                               self.doc_encoder(doc)],
                              1))
