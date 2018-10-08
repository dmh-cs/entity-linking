from typing import Optional
import math
from random import shuffle

from experiment import Experiment
from pyrsistent import m, ny
import pydash as _
import torch
import torch.nn as nn
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss
from torch.utils.data import RandomSampler, BatchSampler, Dataset

import numpy as np

from data_fetchers import get_connection, get_embedding_lookup, get_num_entities, load_page_id_order, load_entity_candidate_ids_and_label_lookup
from default_params import default_train_params, default_model_params, default_run_params, default_paths
from joint_model import JointModel
from logits import Logits
from mention_context_batch_sampler import MentionContextBatchSampler
from mention_context_dataset import MentionContextDataset
from softmax import Softmax
from tester import Tester
from trainer import Trainer
from fire_extinguisher import BatchRepeater
from parsers import parse_for_sentence_spans
from data_transformers import get_mention_sentence_splits, embed_page_content


class BasicDataset(Dataset):
  def __init__(self, data):
    self.data = data
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx]

class Runner(object):
  def __init__(self,
               device,
               paths=default_paths,
               train_params=default_train_params,
               model_params=default_model_params,
               run_params=default_run_params):
    self.train_params = m().update(default_train_params).update(train_params)
    self.model_params = m().update(default_model_params).update(model_params)
    self.run_params = m().update(default_run_params).update(run_params)
    self.paths = m().update(default_paths).update(paths)
    self.experiment = Experiment(self.train_params.update(self.run_params).update(self.model_params))
    self.log = self.experiment.log
    self.lookups = m()
    self.device = device
    self.model_params = self.model_params.set('context_embed_len',
                                              2 * self.model_params.embed_len)
    if not hasattr(self.model_params, 'adaptive_softmax_cutoffs'):
      # self.model_params = self.model_params.set('adaptive_softmax_cutoffs', [100000, 500000])
      # self.model_params = self.model_params.set('adaptive_softmax_cutoffs', [10, 100, 1000, 10000, 100000])
      # self.model_params = self.model_params.set('adaptive_softmax_cutoffs', [20, 25])
      self.model_params = self.model_params.set('adaptive_softmax_cutoffs', [3, 4])
    self.paths = self.paths.set('word_embedding',
                                self._get_word_embedding_path())
    self.page_id_order: Optional[list] = None
    self.num_train_pages: Optional[int] = None
    self.page_id_order_train: Optional[list] = None
    self.page_id_order_test: Optional[list] = None
    self.entity_embeds: Optional[nn.Embedding] = None
    self.adaptive_logits = {'desc': None, 'mention': None}

  def _get_word_embedding_path(self):
    if self.model_params.word_embedding_set.lower() == 'glove' and self.model_params.word_embed_len == 100:
      return'./glove.6B.100d.txt'
    elif self.model_params.word_embedding_set.lower() == 'glove' and self.model_params.word_embed_len == 300:
      return './glove.840B.300d.txt'
    else:
      raise NotImplementedError('Only loading from glove 100d or 300d is currently supported')

  def _get_entity_ids_by_freq(self, cursor):
    query = 'select entity_id, count(*) from entity_mentions group by `entity_id` order by count(*) desc'
    cursor.execute(query)
    sorted_rows = cursor.fetchall()
    return [row['entity_id'] for row in sorted_rows]

  def load_caches(self, cursor):
    self.log.status('Loading word embedding lookup')
    self.lookups = self.lookups.update({'embedding': get_embedding_lookup(self.paths.word_embedding,
                                                                          device=self.device)})
    cursor.execute("select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id = 20388")
    mention_infos = cursor.fetchall()
    entity_ids = list(set([mention_info['entity_id'] for mention_info in mention_infos]))
    mention_infos = [i for i in mention_infos if i['entity_id'] in entity_ids]
    cursor.execute("select content from pages where id = 20388")
    page_content = cursor.fetchone()['content']
    sentence_spans = parse_for_sentence_spans(page_content)
    self.samples = []
    content = ' '.join([mention_info['mention'] for mention_info in mention_infos])
    self.model_params = self.model_params.set('num_entities',
                                              len(set([mention_info['entity_id'] for mention_info in mention_infos])))
    entity_page_mentions = embed_page_content(self.lookups.embedding,
                                              content,
                                              mention_infos)
    entity_labels = {}
    embedded_page_content = embed_page_content(self.lookups.embedding, page_content)
    for mention_info in mention_infos:
      entity_labels[mention_info['entity_id']] = entity_labels[mention_info['entity_id']] if mention_info['entity_id'] in entity_labels else len(entity_labels)
    self.lookups = self.lookups.set('entity_labels',entity_labels)
    for mention_info in mention_infos:
      cands = list(set(entity_labels.values()) - set([entity_labels[mention_info['entity_id']]]))
      shuffle(cands)
      cands = [entity_labels[mention_info['entity_id']]] + cands[:5]
      # cands = list(range(len(entity_labels)))
      # # shuffle(cands)
      self.samples.append({'sentence_splits': get_mention_sentence_splits(page_content,
                                                                          sentence_spans,
                                                                          mention_info),
                           'label': entity_labels[mention_info['entity_id']],
                           'embedded_page_content': embedded_page_content,
                           'entity_page_mentions': entity_page_mentions,
                           # 'candidate_ids': torch.tensor(list(entity_labels.values()))})
                           'candidate_ids': torch.tensor(cands)})
    # shuffle(self.samples)

  def init_entity_embeds(self):
    entity_embed_weights = nn.Parameter(torch.Tensor(self.model_params.num_entities,
                                                     self.model_params.embed_len))
    entity_embed_weights.data.normal_(0, 1.0/math.sqrt(self.model_params.embed_len))
    self.entity_embeds = nn.Embedding(self.model_params.num_entities,
                                      self.model_params.embed_len,
                                      _weight=entity_embed_weights).to(self.device)
    # for p in self.entity_embeds.parameters():
    #   p.requires_grad = False

  def _get_dataset(self, cursor, is_test):
    return BasicDataset(self.samples)

  def _get_sampler(self, cursor, is_test):
    return BatchRepeater(BatchSampler(RandomSampler(list(range(len(self.samples)))),
                                      self.train_params.batch_size,
                                      False))

  def _calc_loss(self, encoded, candidate_entity_ids, labels_for_batch):
    desc_embeds, mention_context_embeds = encoded
    if self.model_params.use_adaptive_softmax:
      desc_logits, desc_loss = self.adaptive_logits['desc'](desc_embeds, labels_for_batch)
      mention_logits, mention_loss = self.adaptive_logits['mention'](mention_context_embeds, labels_for_batch)
    else:
      logits = Logits()
      # weight = np.zeros(candidate_entity_ids.shape[1])
      # counts = np.bincount(labels_for_batch)
      # weight[:len(counts)] = 1.0 / (counts + 0.1)
      # weight = torch.tensor(weight, device=encoded[0].device, dtype=torch.float)
      # criterion = nn.CrossEntropyLoss(weight=weight)
      criterion = nn.CrossEntropyLoss()
      desc_logits = logits(desc_embeds,
                           self.entity_embeds(candidate_entity_ids))
      desc_loss = criterion(desc_logits, labels_for_batch)
      mention_logits = logits(mention_context_embeds,
                              self.entity_embeds(candidate_entity_ids))
      mention_loss = criterion(mention_logits, labels_for_batch)
    return desc_loss + mention_loss

  def _get_trainer(self, cursor, model):
    return Trainer(device=self.device,
                   embedding_lookup=self.lookups.embedding,
                   model=model,
                   get_dataset=lambda: self._get_dataset(cursor, is_test=False),
                   get_batch_sampler=lambda: self._get_sampler(cursor, is_test=False),
                   num_epochs=self.train_params.num_epochs,
                   experiment=self.experiment,
                   calc_loss=self._calc_loss,
                   logits_and_softmax=self._get_logits_and_softmax(),
                   adaptive_logits=self.adaptive_logits,
                   use_adaptive_softmax=self.model_params.use_adaptive_softmax)

  def _get_logits_and_softmax(self):
    def get_calc(context):
      if self.model_params.use_adaptive_softmax:
        softmax = self.adaptive_logits[context].log_prob
        calc = lambda hidden, _: softmax(hidden)
      else:
        calc_logits = Logits()
        softmax = Softmax()
        calc = lambda hidden, candidate_entity_ids: softmax(calc_logits(hidden,
                                                                        self.entity_embeds(candidate_entity_ids)))
      return calc
    return {context: get_calc(context) for context in ['desc', 'mention']}

  def _get_tester(self, cursor, model):
    logits_and_softmax = self._get_logits_and_softmax()
    test_dataset = self._get_dataset(cursor, is_test=True)
    batch_sampler = self._get_sampler(cursor, is_test=True)
    return Tester(dataset=test_dataset,
                  batch_sampler=batch_sampler,
                  model=model,
                  logits_and_softmax=logits_and_softmax,
                  embedding_lookup=self.lookups.embedding,
                  device=self.device,
                  experiment=self.experiment,
                  ablation=self.model_params.ablation,
                  use_adaptive_softmax=self.model_params.use_adaptive_softmax)

  def _get_adaptive_calc_logits(self):
    def get_calc(context):
      if self.model_params.use_hardcoded_cutoffs:
        vocab_size = self.entity_embeds.weight.shape[0]
        cutoffs = self.model_params.adaptive_softmax_cutoffs
      else:
        raise NotImplementedError
      in_features = self.entity_embeds.weight.shape[1]
      n_classes = self.entity_embeds.weight.shape[0]
      return AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=2.0).to(self.device)
    calc = get_calc('')
    return {context: calc for context in ['desc', 'mention']}

  def run(self):
    try:
      db_connection = get_connection()
      with db_connection.cursor() as cursor:
        self.load_caches(cursor)
        pad_vector = self.lookups.embedding['<PAD>']
        self.init_entity_embeds()
        entity_ids_by_freq = self._get_entity_ids_by_freq(cursor)
        self.lookups.set('entity_labels', None)
        self.adaptive_logits = self._get_adaptive_calc_logits()
        encoder = JointModel(self.model_params.embed_len,
                             self.model_params.context_embed_len,
                             self.model_params.word_embed_len,
                             self.model_params.local_encoder_lstm_size,
                             self.model_params.document_encoder_lstm_size,
                             self.model_params.num_lstm_layers,
                             self.train_params.dropout_keep_prob,
                             self.entity_embeds,
                             pad_vector,
                             self.adaptive_logits)
        with self.experiment.train(['mention_context_error', 'document_context_error', 'loss']):
          self.log.status('Training')
          trainer = self._get_trainer(cursor, encoder)
          trainer.train()
          torch.save(encoder.state_dict(), './' + self.experiment.model_name)
    finally:
      db_connection.close()
