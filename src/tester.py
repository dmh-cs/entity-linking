from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import utils as u
from inference import predict

def collate(batch):
  return {'sentence_splits': [sample['sentence_splits'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'embedded_page_content': [sample['embedded_page_content'] for sample in batch],
          'entity_page_mentions': [sample['entity_page_mentions'] for sample in batch],
          'candidates': torch.stack([sample['candidates'] for sample in batch]),
          'p_prior': torch.stack([sample['p_prior'] for sample in batch])}

class Tester(object):
  def __init__(self,
               dataset,
               model,
               entity_embeds,
               embedding_lookup,
               device,
               batch_sampler,
               experiment,
               ablation):
    self.dataset = dataset
    self.model = nn.DataParallel(model)
    self.model = model.to(device)
    self.entity_embeds = entity_embeds
    self.embedding_lookup = embedding_lookup
    self.device = device
    self.batch_sampler = batch_sampler
    self.experiment = experiment
    self.ablation = ablation

  def _get_labels_for_batch(self, labels, candidates):
    device = labels.device
    batch_labels = []
    for label, row_candidates in zip(labels, candidates):
      if label not in row_candidates:
        batch_labels.append(-1)
      else:
        batch_labels.append(int((row_candidates == label).nonzero().squeeze()))
    return torch.tensor(batch_labels, device=device)

  def test(self):
    acc = 0
    n = 0
    dataloader = DataLoader(dataset=self.dataset,
                            batch_sampler=self.batch_sampler,
                            collate_fn=collate)
    for batch_num, batch in enumerate(dataloader):
      batch = u.tensors_to_device(batch, self.device)
      labels_for_batch = self._get_labels_for_batch(batch['label'],
                                                    batch['candidates'])
      predictions = predict(embedding_lookup=self.embedding_lookup,
                            entity_embeds=self.entity_embeds,
                            p_prior=batch['p_prior'],
                            model=self.model,
                            batch=batch,
                            ablation=self.ablation)
      acc += int((labels_for_batch == predictions).sum())
      n += 1
      batch_size = len(predictions)
      self.experiment.record_metrics({'accuracy': acc / (n * batch_size),
                                            'TP': acc,
                                            'num_samples': n * batch_size})
      if batch_num % 100 == 0:
        print(acc, n * batch_size)
    return acc, n
