import torch

def _pad_to_len(coll, max_len, pad_with=None):
  pad_with = pad_with if pad_with is not None else 0
  return coll + [pad_with] * (max_len - len(coll)) if len(coll) < max_len else coll

def collate_bow_doc(bow_doc, device):
  terms = []
  cnts = []
  max_len = 0
  for doc in bow_doc:
    doc_terms = list(doc.keys())
    max_len = max(max_len, len(doc_terms))
    terms.append(doc_terms)
    cnts.append([doc[term] for term in doc_terms])
  terms = torch.tensor([_pad_to_len(doc_terms, max_len) for doc_terms in terms],
                       device=device)
  cnts = torch.tensor([_pad_to_len(doc_term_cnts, max_len, pad_with=0) for doc_term_cnts in cnts],
                      device=device)
  return terms, cnts
