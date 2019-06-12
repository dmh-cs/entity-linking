import torch
import torch.nn as nn

from weighted_sum_encoder import WeightedSumEncoder

from bow_helpers import collate_bow_doc

class EntitySumEncoder(nn.Module):
  def __init__(self, word_embeds, token_ctr_by_entity_id, idf=None):
    super().__init__()
    self.sum_encoder = WeightedSumEncoder(word_embeds, use_cnts=True, idf=idf)
    self.token_ctr_by_entity_id = token_ctr_by_entity_id

  def forward(self, entity_id):
    device = entity_id.device
    ids = sum(entity_id.tolist(), [])
    cntrs = self.token_ctr_by_entity_id(ids)
    return self.sum_encoder(collate_bow_doc(cntrs, device)).reshape(*entity_id.shape, -1)
