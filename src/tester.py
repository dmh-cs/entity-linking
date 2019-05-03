from functools import reduce

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import utils as u
from inference import predict
from data_transformers import pad_batch

def collate_deep_el(batch):
  return {'sentence_splits': [sample['sentence_splits'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'embedded_page_content': [sample['embedded_page_content'] for sample in batch],
          'entity_page_mentions': [sample['entity_page_mentions'] for sample in batch],
          'candidate_ids': torch.stack([sample['candidate_ids'] for sample in batch]),
          'p_prior': torch.stack([sample['p_prior'] for sample in batch]),
          'candidate_mention_sim': torch.stack([torch.tensor(sample['candidate_mention_sim']) for sample in batch])}

def collate_wiki2vec(batch):
  return {'bag_of_nouns': [sample['bag_of_nouns'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'candidate_ids': torch.stack([sample['candidate_ids'] for sample in batch]),
          'p_prior': torch.stack([sample['p_prior'] for sample in batch]),
          'candidate_mention_sim': torch.stack([torch.tensor(sample['candidate_mention_sim']) for sample in batch])}

class Tester(object):
  def __init__(self,
               dataset,
               model,
               logits_and_softmax,
               embedding,
               token_idx_lookup,
               device,
               batch_sampler,
               experiment,
               ablation,
               use_adaptive_softmax,
               use_wiki2vec=False,
               label_to_entity_id=None):
    self.dataset = dataset
    self.model = model.to(device)
    self.embedding = embedding
    self.token_idx_lookup = token_idx_lookup
    self.device = device
    self.batch_sampler = batch_sampler
    self.experiment = experiment
    self.ablation = ablation
    self.logits_and_softmax = logits_and_softmax
    self.use_adaptive_softmax = use_adaptive_softmax
    self.use_wiki2vec = use_wiki2vec
    self.label_to_entity_id = label_to_entity_id

  def _get_labels_for_batch(self, labels, candidate_ids):
    device = labels.device
    batch_labels = []
    for label, row_candidate_ids in zip(labels, candidate_ids):
      if label not in row_candidate_ids:
        batch_labels.append(-1)
      else:
        batch_labels.append(int((row_candidate_ids == label).nonzero().squeeze()))
    return torch.tensor(batch_labels, device=device)

  def test(self):
    if self.use_wiki2vec:
      self.test_wiki2vec()
    else:
      self.test_deep_el()

  def test_deep_el(self):
    acc = 0
    n = 0
    dataloader = DataLoader(dataset=self.dataset,
                            batch_sampler=self.batch_sampler,
                            collate_fn=collate_deep_el)
    for batch_num, batch in enumerate(dataloader):
      batch = u.tensors_to_device(batch, self.device)
      labels_for_batch = self._get_labels_for_batch(batch['label'],
                                                    batch['candidate_ids'])
      predictions = predict(embedding=self.embedding,
                            token_idx_lookup=self.token_idx_lookup,
                            p_prior=0 if self.use_adaptive_softmax else batch['p_prior'],
                            model=self.model,
                            batch=batch,
                            ablation=self.ablation,
                            entity_embeds=self.model.entity_embeds,
                            use_wiki2vec=self.use_wiki2vec)
      missed_idxs = (labels_for_batch != predictions).nonzero().reshape(-1).tolist()
      missed_entity_ids = [self.label_to_entity_id.get(batch['label'][idx]) for idx in missed_idxs]
      missed_sentences = [sum(s.join(' ') for s in batch['sentence_splits']) for idx in missed_idxs]
      guessed_missed = [self.label_to_entity_id.get(predictions[idx]) for idx in missed_idxs]
      with open('./log', 'a') as fh:
        fh.write(reduce(lambda acc, elem: acc + str(elem) + '|',
                        zip(missed_entity_ids, missed_sentences, guessed_missed),
                        ''))
      acc += int((labels_for_batch == predictions).sum())
      batch_size = len(predictions)
      n += batch_size
      self.experiment.record_metrics({'accuracy': acc / n,
                                      'TP': acc,
                                      'num_samples': n})
    return acc, n

  def test_wiki2vec(self):
    acc = 0
    n = 0
    dataloader = DataLoader(dataset=self.dataset,
                            batch_sampler=self.batch_sampler,
                            collate_fn=collate_wiki2vec)
    for batch_num, batch in enumerate(dataloader):
      batch = u.tensors_to_device(batch, self.device)
      labels_for_batch = self._get_labels_for_batch(batch['label'],
                                                    batch['candidate_ids'])
      predictions = predict(embedding=self.embedding,
                            token_idx_lookup=self.token_idx_lookup,
                            p_prior=0 if self.use_adaptive_softmax else batch['p_prior'],
                            model=self.model,
                            batch=batch,
                            ablation=self.ablation,
                            entity_embeds=self.model.entity_embeds,
                            use_wiki2vec=True)
      acc += int((labels_for_batch == predictions).sum())
      batch_size = len(predictions)
      n += batch_size
      self.experiment.record_metrics({'accuracy': acc / n,
                                      'TP': acc,
                                      'num_samples': n})
    return acc, n
