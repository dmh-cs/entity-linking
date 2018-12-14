import pydash as _
import torch

from data_transformers import embed_and_pack_batch
from logits import Logits

def predict(embedding, token_idx_lookup, p_prior, model, batch, ablation, logits_and_softmax):
  model.eval()
  if ablation == ['prior']:
    return torch.argmax(p_prior, dim=1)
  elif 'local_context' in ablation:
    left_splits, right_splits = embed_and_pack_batch(embedding,
                                                     token_idx_lookup,
                                                     batch['sentence_splits'])
    if 'document_context' in ablation:
      mention_embeds, desc_embeds = model.encoder(((left_splits, right_splits),
                                                   batch['embedded_page_content'],
                                                   batch['entity_page_mentions']))
    else:
      local_context = model.local_context_encoder((left_splits, right_splits))
      mention_embeds = model.relu(model.projection(torch.cat((local_context,
                                                              torch.zeros_like(local_context)), 1)))
    calc_logits = Logits()
    men_logits = calc_logits(mention_embeds, batch['candidate_ids'])
    p_text, __ = model.calc_scores((men_logits, torch.zeros_like(men_logits)),
                                   batch['candidate_mention_sim'])
    if 'prior' in ablation:
      posterior = p_prior + p_text - (p_prior * p_text)
      return torch.argmax(posterior, dim=1)
    else:
      return torch.argmax(p_text, dim=1)
  else:
    raise NotImplementedError
