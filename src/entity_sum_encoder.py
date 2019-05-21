import torch
import torch.nn as nn

from weighted_sum_encoder import WeightedSumEncoder

def _pad_to_len(coll, max_len, pad_with=None):
  pad_with = pad_with if pad_with is not None else 0
  return coll + [pad_with] * (max_len - len(coll)) if len(coll) < max_len else coll

def _collate_bow_doc(bow_doc):
  terms = []
  cnts = []
  max_len = 0
  for doc in bow_doc:
    doc_terms = list(doc.keys())
    max_len = max(max_len, len(doc_terms))
    terms.append(doc_terms)
    cnts.append([doc[term] for term in doc_terms])
  terms = torch.tensor([_pad_to_len(doc_terms, max_len) for doc_terms in terms])
  cnts = torch.tensor([_pad_to_len(doc_term_cnts, max_len, pad_with=0) for doc_term_cnts in cnts])
  return terms, cnts

class EntitySumEncoder(nn.Module):
  def __init__(self, word_embeds, token_ctr_by_entity_id):
    super().__init__()
    self.sum_encoder = WeightedSumEncoder(word_embeds, use_cnts=True)
    self.token_ctr_by_entity_id = token_ctr_by_entity_id

  def forward(self, entity_id):
    cntrs = []
    for entity_id in entity_id.tolist():
      cntrs.append(self.token_ctr_by_entity_id[entity_id])
    return self.sum_encoder(_collate_bow_doc(cntrs))
