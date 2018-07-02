from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pydash as _
import itertools
from data_transformers import pad_and_embed_batch

import utils as u

class Trainer:
  def __init__(self, embedding_lookup, model: nn.Module, dataset, num_epochs):
    self.model = model
    self.dataset = dataset
    self.optimizer = self._create_optimizer('adam')
    self.num_epochs = num_epochs
    self.embedding_lookup = embedding_lookup

  def _create_optimizer(self, optimizer: str, params=None):
    print("Creating optimizer '{}' for model:\n{} with params {}".format(optimizer, self.model, params or {}))
    return optim.Adam(self.model.parameters())

  def _classification_error(self, logits, labels):
    predictions = torch.argmax(logits, 1)
    batch_true_labels = torch.arange(len(labels), dtype=torch.long)
    return ((predictions - batch_true_labels) != 0).sum()

  def _get_labels_for_batch(self, labels, candidates):
    return (torch.unsqueeze(labels, 1) == candidates).nonzero()[:, 1]

  def train(self, batch_size):
    for epoch_num in range(self.num_epochs):
      print("Epoch", epoch_num)
      for batch_num, batch in enumerate(self.dataset):
        self.optimizer.zero_grad()
        embedded_sentence_splits = pad_and_embed_batch(self.embedding_lookup,
                                                       batch['sentence_splits'])
        context_embeds = self.model((embedded_sentence_splits,
                                     batch['document_mention_indices']))
        labels_for_batch = self._get_labels_for_batch(batch['label'], batch['candidates'])
        loss = self.model.loss(context_embeds, batch['candidates'], labels_for_batch)
        loss.backward()
        self.optimizer.step()
        if batch_num == 0:
          print(self._classification_error(self.model.logits, batch['label']))
          print('[epoch %d, batch %5d] loss: %.3f' % (epoch_num, batch_num, loss.item()))

    print('Finished Training')
