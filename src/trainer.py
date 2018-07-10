from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pydash as _
import itertools
from data_transformers import pad_and_embed_batch

import utils as u

def collate(batch):
  return {'sentence_splits': [sample['sentence_splits'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'embedded_page_content': [sample['embedded_page_content'] for sample in batch],
          'candidates': torch.stack([sample['candidates'] for sample in batch])}

class Trainer(object):
  def __init__(self,
               embedding_lookup,
               model: nn.Module,
               dataset,
               batch_sampler,
               num_epochs):
    self.model = model
    self.dataset = dataset
    self.batch_sampler = batch_sampler
    self.optimizer = self._create_optimizer('adam')
    self.num_epochs = num_epochs
    self.embedding_lookup = embedding_lookup

  def _create_optimizer(self, optimizer: str, params=None):
    print("Creating optimizer '{}' for model:\n{} with params {}".format(optimizer, self.model, params or {}))
    return optim.Adam(self.model.parameters())

  def _classification_error(self, logits, labels):
    predictions = torch.argmax(logits, 1)
    return int(((predictions - labels) != 0).sum())

  def _get_labels_for_batch(self, labels, candidates):
    return (torch.unsqueeze(labels, 1) == candidates).nonzero()[:, 1]

  def train(self):
    for epoch_num in range(self.num_epochs):
      print("Epoch", epoch_num)
      dataloader = DataLoader(dataset=self.dataset,
                              batch_sampler=self.batch_sampler,
                              collate_fn=collate)
      for batch_num, batch in enumerate(dataloader):
        self.optimizer.zero_grad()
        embedded_sentence_splits = pad_and_embed_batch(self.embedding_lookup,
                                                       batch['sentence_splits'])
        encoded = self.model((embedded_sentence_splits,
                              batch['embedded_page_content']))
        labels_for_batch = self._get_labels_for_batch(batch['label'], batch['candidates'])
        loss = self.model.loss(encoded, batch['candidates'], labels_for_batch)
        loss.backward()
        self.optimizer.step()
        # if batch_num % 100 == 0:
        if True:
          print('Classification error',
                self._classification_error(self.model.mention_context_encoder.logits,
                                           labels_for_batch))
          print('Classification error',
                self._classification_error(self.model.desc_encoder.logits,
                                           labels_for_batch))
          print('[epoch %d, batch %d] loss: %.3f' % (epoch_num, batch_num, loss.item()))
    print('Finished Training')
