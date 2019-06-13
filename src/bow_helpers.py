import torch
from torch.nn.utils.rnn import pad_sequence

def collate_bow_doc(bow_doc, device):
  terms = []
  cnts = []
  for doc in bow_doc:
    doc_terms = list(doc.keys())
    terms.append(torch.tensor(doc_terms))
    cnts.append(torch.tensor([doc[term] for term in doc_terms]))
  terms = pad_sequence(terms, batch_first=True).to(device)
  cnts = pad_sequence(cnts, batch_first=True).to(device)
  return terms, cnts
