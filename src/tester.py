import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

from data_transformers import embed_and_pack_batch
import utils as u

def collate(batch):
  return {'sentence_splits': [sample['sentence_splits'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'candidates': torch.stack([sample['candidates'] for sample in batch])}

class Tester(object):
  def __init__(self, dataset, model, entity_embeds, embedding_lookup, device, batch_sampler):
    self.dataset = dataset
    self.model = nn.DataParallel(model)
    self.model = model.to(device)
    self.entity_embeds = entity_embeds
    self.embedding_lookup = embedding_lookup
    self.device = device
    self.batch_sampler = batch_sampler

  def _get_labels_for_batch(self, labels, candidates):
    return (torch.unsqueeze(labels, 1) == candidates).nonzero()[:, 1]

  def test(self):
    acc = 0
    n = 0
    dataloader = DataLoader(dataset=self.dataset,
                            batch_sampler=self.batch_sampler,
                            collate_fn=collate)
    for batch in dataloader:
      batch = u.tensors_to_device(batch, self.device)
      left_splits, right_splits = embed_and_pack_batch(self.embedding_lookup,
                                                       batch['sentence_splits'])
      mention_embeds = self.model((left_splits, right_splits))
      labels_for_batch = self._get_labels_for_batch(batch['label'],
                                                    batch['candidates'])
      logits = torch.sum(torch.mul(torch.unsqueeze(mention_embeds, 1),
                                   self.entity_embeds(batch['candidates'])),
                         2)
      predictions = torch.argmax(logits, dim=1)
      acc += (labels_for_batch == predictions).sum()
      n += 1
    return acc, n
