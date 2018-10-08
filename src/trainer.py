import itertools

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import pydash as _

from data_transformers import embed_and_pack_batch

from utils import tensors_to_device

def collate(batch):
  return {'sentence_splits': [sample['sentence_splits'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'embedded_page_content': [sample['embedded_page_content'] for sample in batch],
          'entity_page_mentions': [sample['entity_page_mentions'] for sample in batch],
          'candidate_ids': torch.stack([sample['candidate_ids'] for sample in batch])}

class Trainer(object):
  def __init__(self,
               device,
               embedding,
               token_idx_lookup,
               model: nn.Module,
               get_dataset,
               get_batch_sampler,
               num_epochs,
               experiment,
               calc_loss,
               logits_and_softmax,
               adaptive_logits,
               use_adaptive_softmax):
    self.device = device
    self.model = nn.DataParallel(model)
    self.model = model.to(self.device)
    self.get_dataset = get_dataset
    self.get_batch_sampler = get_batch_sampler
    self.num_epochs = num_epochs
    self.embedding = embedding
    self.token_idx_lookup = token_idx_lookup
    self.experiment = experiment
    self.calc_loss = calc_loss
    self.logits_and_softmax = logits_and_softmax
    self.adaptive_logits = adaptive_logits
    self.optimizer = self._create_optimizer('adam')
    self.use_adaptive_softmax = use_adaptive_softmax

  def _create_optimizer(self, optimizer: str, params=None):
    return optim.Adam(itertools.chain(self.model.parameters(),
                                      self.adaptive_logits['desc'].parameters(),
                                      self.adaptive_logits['mention'].parameters()))

  def _classification_error(self, logits, labels):
    predictions = torch.argmax(logits, 1)
    return int(((predictions - labels) != 0).sum())

  def _get_labels_for_batch(self, labels, candidate_ids):
    return (torch.unsqueeze(labels, 1) == candidate_ids).nonzero()[:, 1]

  def train(self):
    for epoch_num in range(self.num_epochs):
      self.experiment.update_epoch(epoch_num)
      dataloader = DataLoader(dataset=self.get_dataset(),
                              batch_sampler=self.get_batch_sampler(),
                              collate_fn=collate)
      for batch_num, batch in enumerate(dataloader):
        self.optimizer.zero_grad()
        batch = tensors_to_device(batch, self.device)
        if self.use_adaptive_softmax:
          labels = batch['label']
        else:
          labels = self._get_labels_for_batch(batch['label'], batch['candidate_ids'])
        left_splits, right_splits = embed_and_pack_batch(self.embedding,
                                                         self.token_idx_lookup,
                                                         batch['sentence_splits'])
        encoded = self.model(((left_splits, right_splits),
                              batch['embedded_page_content'],
                              batch['entity_page_mentions']))
        desc_embeds, mention_embeds = encoded
        loss = self.calc_loss(encoded, batch['candidate_ids'], labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(itertools.chain(self.model.parameters(),
        #                                                self.adaptive_logits['desc'].parameters(),
        #                                                self.adaptive_logits['mention'].parameters()),
        #                                0.01)
        self.optimizer.step()
        with torch.no_grad():
          mention_probas = self.logits_and_softmax['mention'](mention_embeds, batch['candidate_ids'])
          desc_probas = self.logits_and_softmax['desc'](desc_embeds, batch['candidate_ids'])
          mention_context_error = self._classification_error(mention_probas, labels)
          document_context_error = self._classification_error(desc_probas, labels)
        self.experiment.record_metrics({'mention_context_error': mention_context_error,
                                        'document_context_error': document_context_error,
                                        'loss': loss.item()},
                                       batch_num=batch_num)
