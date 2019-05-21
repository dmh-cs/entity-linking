import pydash as _
import torch
from collections import Counter

from data_transformers import embed_and_pack_batch
from logits import Logits
from utils import to_idx

def predict(embedding, token_idx_lookup, p_prior, model, batch, ablation, entity_embeds, use_wiki2vec=False, use_stacker=True, use_sum_encoder=False):
  if use_wiki2vec:
    return predict_wiki2vec(embedding, token_idx_lookup, p_prior, model, batch, ablation, entity_embeds)
  elif use_sum_encoder:
    return predict_sum_encoder(embedding, token_idx_lookup, p_prior, model, batch, ablation, entity_embeds, use_stacker)
  else:
    return predict_deep_el(embedding, token_idx_lookup, p_prior, model, batch, ablation, entity_embeds, use_stacker)

def predict_wiki2vec(embedding, token_idx_lookup, p_prior, model, batch, ablation, entity_embeds):
  model.eval()
  context = model.encoder(batch['bag_of_nouns'])
  logits = Logits()
  calc_logits = lambda embeds, ids: logits(embeds, entity_embeds(ids))
  context_logits = calc_logits(context, batch['candidate_ids'])
  p_text, __ = model.calc_scores((context_logits, torch.zeros_like(context_logits)),
                                 batch['candidate_mention_sim'],
                                 p_prior)
  return torch.argmax(p_text, dim=1)

def predict_deep_el(embedding, token_idx_lookup, p_prior, model, batch, ablation, entity_embeds, use_stacker):
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
      mention_embeds = model.encoder.mention_context_encoder(((left_splits, right_splits),
                                                             batch['embedded_page_content'],
                                                             batch['entity_page_mentions']))
    logits = Logits()
    calc_logits = lambda embeds, ids: logits(embeds, entity_embeds(ids))
    men_logits = calc_logits(mention_embeds, batch['candidate_ids'])
    if use_stacker:
      p_text, __ = model.calc_scores((men_logits, torch.zeros_like(men_logits)),
                                     batch['candidate_mention_sim'],
                                     p_prior)
    else:
      p_text = men_logits
    return torch.argmax(p_text, dim=1)
  else:
    raise NotImplementedError

def predict_sum_encoder(embedding, token_idx_lookup, p_prior, model, batch, ablation, entity_embeds, use_stacker):
  model.eval()
  context_bows = [Counter(to_idx(token_idx_lookup, token) for token in sentence)
                        for sentence in batch['mention_sentence']]
  doc_bows = batch['page_token_cnts']
  encoded = model.encoder(context_bows, doc_bows)
  logits = Logits()
  calc_logits = lambda embeds, ids: logits(embeds, entity_embeds(ids))
  men_logits = calc_logits(encoded, batch['candidate_ids'])
  if use_stacker:
    p_text, __ = model.calc_scores((men_logits, torch.zeros_like(men_logits)),
                                   batch['candidate_mention_sim'],
                                   p_prior)
  else:
    p_text = men_logits
  return torch.argmax(p_text, dim=1)
