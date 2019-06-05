import torch
import torch.nn as nn

from weighted_sum_encoder import WeightedSumEncoder

from bow_helpers import collate_bow_doc

class EntitySumEncoder(nn.Module):
  def __init__(self, word_embeds, token_ctr_by_entity_id):
    super().__init__()
    self.sum_encoder = WeightedSumEncoder(word_embeds, use_cnts=True)
    self.token_ctr_by_entity_id = token_ctr_by_entity_id

  def forward(self, entity_id):
    device = entity_id.device
    cntrs = []
    for batch_entity_ids in entity_id.tolist():
      for single_id in batch_entity_ids:
        cntrs.append(self.token_ctr_by_entity_id.get(single_id, {1: 1}))
    return self.sum_encoder(collate_bow_doc(cntrs, device)).reshape(*entity_id.shape, -1)
