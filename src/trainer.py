from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pydash as _
import itertools

import utils as u

class Trainer:
  def __init__(self, model: nn.Module, datasets):
    self.model = model
    self.datasets = datasets
    self.optimizer = self._create_optimizer('adam')
    self.num_epochs = 1000

  def _create_optimizer(self, optimizer: str, params=None):
    print("Creating optimizer '{}' for model:\n{} with params {}".format(optimizer, self.model, params or {}))
    return optim.Adam(self.model.parameters())

  def train(self):
    for epoch_num in range(self.num_epochs):
      print("Epoch", epoch_num)
      for batch_num, batch in enumerate(self.datasets['train']):
        self.optimizer.zero_grad()
        desc_embeds = self.model(torch.unsqueeze(batch['description'], 1))
        loss = self.model.loss(desc_embeds, batch['label'])
        print(((torch.argmax(self.model.logits, 1) - batch['label']) != 0).sum())
        print(torch.mm(desc_embeds, torch.transpose(self.model.entity_embeds.weight, 0, 1)))
        loss.backward()
        self.optimizer.step()
        print('[epoch %d, batch %5d] loss: %.3f' %
              (epoch_num, batch_num, loss.item()))

    print('Finished Training')
