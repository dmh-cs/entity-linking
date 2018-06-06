from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pydash as _

import utils as u

class Trainer:
  def __init__(self, model: nn.Module, raw_datasets):
    self.model = model
    self.raw_datasets = raw_datasets
    self.optimizer = self._create_optimizer('adam')
    self.batch_size = 1000

  def _create_optimizer(self, optimizer: str, params=None):
    print("Creating optimizer '{}' for model:\n{} with params {}".format(optimizer, self.model, params or {}))
    return optim.Adam(self.model.parameters())

  def train(self):
    train_data_batches = u.get_batches(self.raw_datasets['train'], self.batch_size)
    for epoch in range(2):
      running_loss = 0.0
      for i, data in enumerate(train_data_batches):
        descriptions, entity_ids = data
        self.optimizer.zero_grad()
        desc_embeds = self.model(descriptions)
        loss = self.model.loss(desc_embeds, entity_ids)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 0:
          print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
          running_loss = 0.0

    print('Finished Training')
