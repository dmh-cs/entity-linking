from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pydash as _
import itertools
from data_transformers import embed_and_pack_batch

from utils import tensors_to_device, collate

class Trainer(object):
  def __init__(self,
               device,
               embedding_lookup,
               model: nn.Module,
               dataset,
               batch_sampler,
               num_epochs):
    self.device = device
    self.model = nn.DataParallel(model)
    self.model = model.to(self.device)
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
        batch = tensors_to_device(batch, self.device)
        self.optimizer.zero_grad()
        left_splits, right_splits = embed_and_pack_batch(self.embedding_lookup,
                                                         batch['sentence_splits'])
        encoded = self.model(((left_splits, right_splits),
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
