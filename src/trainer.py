import itertools
from collections import Counter

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import pydash as _

from data_transformers import embed_and_pack_batch

from utils import tensors_to_device, to_idx

def collate_deep_el(batch):
  return {'sentence_splits': [sample['sentence_splits'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'embedded_page_content': [sample['embedded_page_content'] for sample in batch],
          'entity_page_mentions': [sample['entity_page_mentions'] for sample in batch],
          'candidate_ids': torch.stack([sample['candidate_ids'] for sample in batch]),
          'prior': torch.stack([sample['p_prior'] for sample in batch]),
          'candidate_mention_sim': torch.stack([sample['candidate_mention_sim'] for sample in batch])}

def collate_sum_encoder(batch):
  return {'mention_sentence': [sample['mention_sentence'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'page_token_cnts': [sample['page_token_cnts'] for sample in batch],
          'candidate_ids': torch.stack([sample['candidate_ids'] for sample in batch]),
          'prior': torch.stack([sample['p_prior'] for sample in batch]),
          'candidate_mention_sim': torch.stack([sample['candidate_mention_sim'] for sample in batch])}

def collate_wiki2vec(batch):
  return {'bag_of_nouns': [sample['bag_of_nouns'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'candidate_ids': torch.stack([sample['candidate_ids'] for sample in batch]),
          'prior': torch.stack([sample['p_prior'] for sample in batch]),
          'candidate_mention_sim': torch.stack([sample['candidate_mention_sim'] for sample in batch])}

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
               calc_logits,
               logits_and_softmax,
               adaptive_logits,
               use_adaptive_softmax,
               clip_grad,
               use_wiki2vec=False,
               use_sum_encoder=False,
               use_stacker=True,
               dont_clip_grad=False):
    self.device = device
    self.model = model.to(self.device)
    self.get_dataset = get_dataset
    self.get_batch_sampler = get_batch_sampler
    self.num_epochs = num_epochs
    self.embedding = embedding
    self.token_idx_lookup = token_idx_lookup
    self.experiment = experiment
    self.calc_loss = calc_loss
    self.calc_logits = calc_logits
    self.logits_and_softmax = logits_and_softmax
    self.adaptive_logits = adaptive_logits
    self.optimizer = self._create_optimizer('adam')
    self.use_adaptive_softmax = use_adaptive_softmax
    self.clip_grad = clip_grad
    self.use_wiki2vec = use_wiki2vec
    self.use_sum_encoder = use_sum_encoder
    self.use_stacker = use_stacker
    self.dont_clip_grad = dont_clip_grad

  def _get_adaptive_logits_params(self):
    if self.adaptive_logits['desc'] is not None:
      return itertools.chain(self.adaptive_logits['desc'].parameters(),
                             self.adaptive_logits['mention'].parameters())
    else:
      return []

  def _create_optimizer(self, optimizer: str, params=None):
    return optim.Adam(itertools.chain(self.model.parameters(),
                                      self._get_adaptive_logits_params()))

  def _classification_error(self, logits, labels):
    predictions = torch.argmax(logits, 1)
    return int(((predictions - labels) != 0).sum())

  def _get_labels_for_batch(self, labels, candidate_ids):
    device = labels.device
    batch_labels = []
    for label, row_candidate_ids in zip(labels, candidate_ids):
      if label not in row_candidate_ids:
        batch_labels.append(-1)
      else:
        batch_labels.append(int((row_candidate_ids == label).nonzero().squeeze()))
    return torch.tensor(batch_labels, device=device)

  def train(self):
    if self.use_wiki2vec:
      self.train_wiki2vec()
    elif self.use_sum_encoder:
      self.train_sum_encoder()
    else:
      self.train_deep_el()

  def train_deep_el(self):
    for epoch_num in range(self.num_epochs):
      self.experiment.update_epoch(epoch_num)
      self._dataset = self.get_dataset()
      dataloader = DataLoader(dataset=self._dataset,
                              batch_sampler=self.get_batch_sampler(),
                              collate_fn=collate_deep_el)
      for batch_num, batch in enumerate(dataloader):
        if self._dataset.use_fast_sampler:
          dataloader.batch_sampler.page_ctr = dataloader.dataset.page_ctr
        self.model.train()
        self.optimizer.zero_grad()
        batch = tensors_to_device(batch, self.device)
        if self.use_adaptive_softmax:
          labels = batch['label']
        else:
          labels = self._get_labels_for_batch(batch['label'], batch['candidate_ids'])
        left_splits, right_splits = embed_and_pack_batch(self.embedding,
                                                         self.token_idx_lookup,
                                                         batch['sentence_splits'])
        encoded = self.model.encoder(((left_splits, right_splits),
                                      batch['embedded_page_content'],
                                      batch['entity_page_mentions']))
        logits = self.calc_logits(encoded, batch['candidate_ids'])
        if self.use_stacker:
          scores = self.model.calc_scores(logits,
                                          batch['candidate_mention_sim'],
                                          batch['prior'])
        else:
          scores = logits
        scores = scores[(labels != -1).nonzero().reshape(-1)]
        labels = labels[(labels != -1).nonzero().reshape(-1)]
        loss = self.calc_loss(scores, labels)
        loss.backward()
        if not self.dont_clip_grad:
          torch.nn.utils.clip_grad_norm_(itertools.chain(self.model.parameters(),
                                                         self._get_adaptive_logits_params()),
                                         self.clip_grad)
        self.optimizer.step()
        with torch.no_grad():
          self.model.eval()
          encoded_test = self.model.encoder(((left_splits, right_splits),
                                             batch['embedded_page_content'],
                                             batch['entity_page_mentions']))
          logits_test = self.calc_logits(encoded_test, batch['candidate_ids'])
          if self.use_stacker:
            desc_probas, mention_probas = self.model.calc_scores(logits_test,
                                                                 batch['candidate_mention_sim'],
                                                                 batch['prior'])
          else:
            desc_probas, mention_probas = logits_test
          mention_context_error = self._classification_error(mention_probas, labels)
          document_context_error = self._classification_error(desc_probas, labels)
        self.experiment.record_metrics({'mention_context_error': mention_context_error,
                                        'document_context_error': document_context_error,
                                        'loss': loss.item()},
                                       batch_num=batch_num)
      torch.save(self.model.state_dict(), './' + self.experiment.model_name + '_epoch_' + str(epoch_num))

  def train_sum_encoder(self):
    for epoch_num in range(self.num_epochs):
      self.experiment.update_epoch(epoch_num)
      self._dataset = self.get_dataset()
      dataloader = DataLoader(dataset=self._dataset,
                              batch_sampler=self.get_batch_sampler(),
                              collate_fn=collate_sum_encoder)
      for batch_num, batch in enumerate(dataloader):
        if self._dataset.use_fast_sampler:
          dataloader.batch_sampler.page_ctr = dataloader.dataset.page_ctr
        self.model.train()
        self.optimizer.zero_grad()
        batch = tensors_to_device(batch, self.device)
        labels = self._get_labels_for_batch(batch['label'], batch['candidate_ids'])
        context_bows = [Counter(to_idx(self.token_idx_lookup, token) for token in sentence)
                        for sentence in batch['mention_sentence']]
        doc_bows = batch['page_token_cnts']
        encoded = self.model.encoder(context_bows, doc_bows)
        logits = self.calc_logits(encoded, batch['candidate_ids'])
        if self.use_stacker:
          scores = self.model.calc_scores(logits,
                                          batch['candidate_mention_sim'],
                                          batch['prior'])
        else:
          scores = logits
        scores = scores[(labels != -1).nonzero().reshape(-1)]
        labels = labels[(labels != -1).nonzero().reshape(-1)]
        loss = self.calc_loss(scores, labels)
        loss.backward()
        if not self.dont_clip_grad:
          torch.nn.utils.clip_grad_norm_(itertools.chain(self.model.parameters(),
                                                         self._get_adaptive_logits_params()),
                                         self.clip_grad)
        self.optimizer.step()
        with torch.no_grad():
          self.model.eval()
          encoded_test = self.model.encoder(context_bows, doc_bows)
          logits_test = self.calc_logits(encoded_test, batch['candidate_ids'])
          if self.use_stacker:
            probas = self.model.calc_scores(logits_test,
                                            batch['candidate_mention_sim'],
                                            batch['prior'])
          else:
            probas = logits_test
          context_error = self._classification_error(probas, labels)
        self.experiment.record_metrics({'error': context_error,
                                        'loss': loss.item()},
                                       batch_num=batch_num)
      torch.save(self.model.state_dict(), './' + self.experiment.model_name + '_epoch_' + str(epoch_num))

  def train_wiki2vec(self):
    for epoch_num in range(self.num_epochs):
      self.experiment.update_epoch(epoch_num)
      self._dataset = self.get_dataset()
      dataloader = DataLoader(dataset=self._dataset,
                              batch_sampler=self.get_batch_sampler(),
                              collate_fn=collate_wiki2vec)
      for batch_num, batch in enumerate(dataloader):
        self.model.train()
        self.optimizer.zero_grad()
        batch = tensors_to_device(batch, self.device)
        labels = self._get_labels_for_batch(batch['label'], batch['candidate_ids'])
        encoded = self.model.encoder(batch['bag_of_nouns'])
        logits = self.calc_logits(encoded, batch['candidate_ids'])
        scores = self.model.calc_scores((logits, torch.zeros_like(logits)),
                                        batch['candidate_mention_sim'],
                                        batch['prior'])
        loss = self.calc_loss(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(itertools.chain(self.model.parameters(),
                                                       self._get_adaptive_logits_params()),
                                       self.clip_grad)
        self.optimizer.step()
        with torch.no_grad():
          self.model.eval()
          encoded_test = self.model.encoder(batch['bag_of_nouns'])
          logits_test = self.calc_logits(encoded_test, batch['candidate_ids'])
          mention_probas, __ = self.model.calc_scores((logits_test, torch.zeros_like(logits_test)),
                                                      batch['candidate_mention_sim'],
                                                      batch['prior'])
          context_error = self._classification_error(mention_probas, labels)
        self.experiment.record_metrics({'error': context_error,
                                        'loss': loss.item()},
                                       batch_num=batch_num)
