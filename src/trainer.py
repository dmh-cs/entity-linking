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
    self.batch_size = 10
    self.num_epochs = 2

  def _create_optimizer(self, optimizer: str, params=None):
    print("Creating optimizer '{}' for model:\n{} with params {}".format(optimizer, self.model, params or {}))
    return optim.Adam(self.model.parameters())

  def train(self):
    train_descriptions, train_labels = self.datasets['train']
    epochs = zip(itertools.tee(train_descriptions, self.num_epochs),
                 itertools.tee(train_labels, self.num_epochs))
    for epoch_num, epoch_data in enumerate(epochs):
      print("Epoch", epoch_num)
      descriptions, labels = epoch_data
      descriptions_batches = u.get_batches(descriptions, self.batch_size)
      labels_batches = u.get_batches(labels, self.batch_size)
      running_loss = 0.0
      for batch_num, batch in enumerate(zip(descriptions_batches, labels_batches)):
        descriptions_batch, labels_batch = batch
        self.optimizer.zero_grad()
        desc_embeds = self.model(torch.unsqueeze(descriptions_batch, 1))
        loss = self.model.loss(desc_embeds, labels_batch)
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
        if batch_num % 10 == 9:
          print('[epoch %d, batch %5d] loss: %.3f' %
              (epoch_num, batch_num, running_loss / 10))
          running_loss = 0.0

    print('Finished Training')
