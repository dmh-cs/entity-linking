import torch
from sklearn.metrics import confusion_matrix, accuracy_score

from data_transformers import pad_and_embed_batch

class Tester(object):
  def __init__(self, dataset, model, entity_embeds, embedding_lookup):
    self.dataset = dataset
    self.model = model
    self.entity_embeds = entity_embeds
    self.embedding_lookup = embedding_lookup

  def _get_labels_for_batch(self, labels, candidates):
    return (torch.unsqueeze(labels, 1) == candidates).nonzero()[:, 1]

  def test(self):
    cm = torch.tensor([[0, 0],
                       [0, 0]])
    for elem in self.dataset:
      embedded_sentence_splits = pad_and_embed_batch(self.embedding_lookup,
                                                     [elem['sentence_splits']])
      _, mention_embeds = self.model((embedded_sentence_splits,
                                      torch.unsqueeze(elem['embedded_page_content'], 0)))
      labels_for_batch = self._get_labels_for_batch(torch.tensor([elem['label']]),
                                                    torch.unsqueeze(elem['candidates'], 0))
      logits = torch.sum(torch.mul(torch.unsqueeze(mention_embeds, 1),
                                   self.entity_embeds(torch.unsqueeze(elem['candidates'], 0))),
                         2)
      predictions = torch.argmax(logits, dim=1)
      print(accuracy_score(labels_for_batch, predictions))
      cm += torch.tensor(confusion_matrix(labels_for_batch, predictions))
    return cm
