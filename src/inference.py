import pydash as _
import torch

from data_transformers import embed_and_pack_batch

def predict(embedding_lookup, entity_embeds, p_prior, model, batch, ablation, logits_and_softmax):
  if ablation == ['prior']:
    return torch.argmax(p_prior, dim=1)
  elif 'local_context' in ablation:
    left_splits, right_splits = embed_and_pack_batch(embedding_lookup,
                                                     batch['sentence_splits'])
    if 'document_context' in ablation:
      _, mention_embeds = model(((left_splits, right_splits),
                                 batch['embedded_page_content'],
                                 batch['entity_page_mentions']))
    else:
      local_context = model.local_context_encoder((left_splits, right_splits))
      mention_embeds = model.relu(model.projection(torch.cat((local_context,
                                                              torch.zeros_like(local_context)), 1)))
    p_text = logits_and_softmax(mention_embeds, entity_embeds(batch['candidates']))
    if 'prior' in ablation:
      posterior = p_prior + p_text - (p_prior * p_text)
      return torch.argmax(posterior, dim=1)
    else:
      return torch.argmax(p_text, dim=1)
  else:
    raise NotImplementedError
