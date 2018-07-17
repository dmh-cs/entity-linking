import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score

from data_transformers import embed_and_pack_batch
from utils import tensors_to_device

class Tester(object):
  def __init__(self, dataset, model, entity_embeds, embedding_lookup, device):
    self.dataset = dataset
    self.model = nn.DataParallel(model)
    self.model = model.to(device)
    self.entity_embeds = entity_embeds
    self.embedding_lookup = embedding_lookup
    self.device = device

  def _get_labels_for_batch(self, labels, candidates):
    return (torch.unsqueeze(labels, 1) == candidates).nonzero()[:, 1]

  def test(self):
    acc = 0
    n = 0
    for elem in self.dataset:
      elem = tensors_to_device(elem, self.device)
      left_splits, right_splits = embed_and_pack_batch(self.embedding_lookup,
                                                       [elem['sentence_splits']])
      _, mention_embeds = self.model(((left_splits, right_splits),
                                      torch.unsqueeze(elem['embedded_page_content'], 0)))
      labels_for_batch = self._get_labels_for_batch(torch.unsqueeze(elem['label'], 0),
                                                    torch.unsqueeze(elem['candidates'], 0))
      logits = torch.sum(torch.mul(torch.unsqueeze(mention_embeds, 1),
                                   self.entity_embeds(torch.unsqueeze(elem['candidates'], 0))),
                         2)
      predictions = torch.argmax(logits, dim=1)
      acc += (labels_for_batch == predictions).sum()
      n += 1
    return acc, n
