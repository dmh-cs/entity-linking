import pickle
from typing import Optional
import math

import pydash as _
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, RandomSampler
from pyrsistent import m

from data_fetchers import get_connection, get_embedding_lookup
from joint_model import JointModel
from simple_mention_context_dataset_by_entity_ids import SimpleMentionContextDatasetByEntityIds
from mention_context_dataset import MentionContextDataset
from mention_context_batch_sampler import MentionContextBatchSampler
from trainer import Trainer
from tester import Tester

def load_entity_candidates_and_label_lookup(path):
  with open(path, 'rb') as lookup_file:
    return pickle.load(lookup_file)

def load_page_id_order(path):
  with open(path, 'rb') as f:
    return pickle.load(f)

def get_num_entities():
  try:
    db_connection = get_connection()
    with db_connection.cursor() as cursor:
      cursor.execute('select count(*) from entities')
      return cursor.fetchone()['count(*)']
  finally:
    db_connection.close()

default_paths = m(lookups_path='../entity-linking-preprocessing/lookups.pkl',
                  page_id_order_path='../entity-linking-preprocessing/page_id_order.pkl')

default_train_params = m(batch_size=50,
                         load_model=False,
                         debug=False,
                         use_simple_dataloader=False,
                         num_epochs=1,
                         train_size=0.8,
                         dropout_keep_prob=0.5)

default_model_params = m(embed_len=100,
                         word_embed_len=100,
                         num_candidates=10,
                         word_embedding_set='glove',
                         lstm_size=100,
                         num_lstm_layers=2)

class Runner(object):
  def __init__(self,
               device,
               paths=default_paths,
               train_params=default_train_params,
               model_params=default_model_params):
    self.batch_size:               Optional[int] = None
    self.num_entities:             Optional[int] = None
    self.load_model:               Optional[bool] = None
    self.debug:                    Optional[bool] = None
    self.use_simple_dataloader:    Optional[bool] = None
    self.num_epochs:               Optional[int] = None
    self.train_size:               Optional[float] = None
    self.embed_len:                Optional[int] = None
    self.word_embed_len:           Optional[int] = None
    self.num_candidates:           Optional[int] = None
    self.word_embedding_set:       Optional[str] = None
    self.lookups_path:             Optional[str] = None
    self.page_id_order_path:       Optional[str] = None
    self.entity_candidates_lookup: Optional[dict] = None
    self.entity_label_lookup:      Optional[dict] = None
    self.embedding_lookup:         Optional[dict] = None
    self.page_id_order:            Optional[list] = None
    self.num_train_pages:          Optional[int] = None
    self.page_id_order_train:      Optional[list] = None
    self.page_id_order_test:       Optional[list] = None
    self.entity_embeds:            Optional[nn.Embedding] = None
    self.dropout_keep_prob:        Optional[float] = None
    self.lstm_size:                Optional[int] = None
    self.num_lstm_layers:          Optional[int] = None
    self.entity_ids_for_simple_dataset: Optional[list] = None
    self.device = device

    _.defaults(self,
               default_model_params, default_train_params, default_paths,
               train_params, model_params, paths)
    self.context_embed_len = 2 * self.embed_len
    self.word_embedding_path = self._get_word_embedding_path()
    if not self.use_simple_dataloader and self.num_entities is not None:
      raise NotImplementedError('Can only restrict num of entities when using simple dataloader')

  def _get_word_embedding_path(self):
    if self.word_embedding_set.lower() == 'glove' and self.word_embed_len == 100:
      return'./glove.6B.100d.txt'
    else:
      raise NotImplementedError('Only loading from glove 100d is currently supported')

  def load_caches(self):
    if self.num_entities is None:
      log.status('Getting number of entities')
      self.num_entities = get_num_entities()
    log.status('Loading entity candidates lookup')
    lookups = load_entity_candidates_and_label_lookup(self.lookups_path)
    self.entity_candidates_lookup = lookups['entity_candidates']
    self.entity_label_lookup = lookups['entity_labels']
    log.status('Loading word embedding lookup')
    self.embedding_lookup = get_embedding_lookup(self.word_embedding_path, device=self.device)
    log.status('Getting page id order')
    self.page_id_order = load_page_id_order(self.page_id_order_path)
    self.num_train_pages = int(len(self.page_id_order) * self.train_size)
    self.page_id_order_train = self.page_id_order[:self.num_train_pages]
    self.page_id_order_test = self.page_id_order[self.num_train_pages:]
    if self.use_simple_dataloader:
      self.entity_ids_for_simple_dataset = list(sorted(self.entity_label_lookup.keys()))[:self.num_entities]

  def init_entity_embeds(self):
    entity_embed_weights = nn.Parameter(torch.Tensor(self.num_entities, self.embed_len))
    entity_embed_weights.data.normal_(0, 1.0/math.sqrt(self.embed_len))
    self.entity_embeds = nn.Embedding(self.num_entities, self.embed_len, _weight=entity_embed_weights)

  def _get_simple_dataset(self, cursor, is_test):
    return SimpleMentionContextDatasetByEntityIds(cursor,
                                                  self.entity_candidates_lookup,
                                                  self.entity_label_lookup,
                                                  self.embedding_lookup,
                                                  self.num_candidates,
                                                  self.entity_ids_for_simple_dataset,
                                                  is_test)

  def _get_dataset(self, cursor, is_test):
    page_ids = self.page_id_order_test if is_test else self.page_id_order_train
    return MentionContextDataset(cursor,
                                 page_ids,
                                 self.entity_candidates_lookup,
                                 self.entity_label_lookup,
                                 self.embedding_lookup,
                                 self.batch_size,
                                 self.num_entities,
                                 self.num_candidates)

  def _get_sampler(self, cursor, is_test):
    page_ids = self.page_id_order_test if is_test else self.page_id_order_train
    return MentionContextBatchSampler(cursor,
                                      page_ids,
                                      self.batch_size)

  def _get_trainer(self, cursor, model):
    if self.use_simple_dataloader:
      train_dataset = self._get_simple_dataset(cursor, is_test=False)
      batch_sampler = BatchSampler(RandomSampler(train_dataset), self.batch_size, True)
    else:
      train_dataset = self._get_dataset(cursor, is_test=False)
      batch_sampler = self._get_sampler(cursor, is_test=False)
    return Trainer(device=self.device,
                   embedding_lookup=self.embedding_lookup,
                   model=model,
                   dataset=train_dataset,
                   batch_sampler=batch_sampler,
                   num_epochs=self.num_epochs)

  def _get_tester(self, cursor, model):
    if self.use_simple_dataloader:
      test_dataset = self._get_simple_dataset(cursor, is_test=True)
      batch_sampler = BatchSampler(RandomSampler(test_dataset), self.batch_size, True)
    else:
      test_dataset = self._get_dataset(cursor, is_test=True)
      batch_sampler = self._get_sampler(cursor, is_test=True)
    return Tester(dataset=test_dataset,
                        batch_sampler=batch_sampler,
                        model=model.module.mention_context_encoder,
                        entity_embeds=self.entity_embeds,
                        embedding_lookup=self.embedding_lookup,
                        device=self.device)

  def run(self):
    pad_vector = self.embedding_lookup['<PAD>']
    self.init_entity_embeds()
    try:
      db_connection = get_connection()
      with db_connection.cursor() as cursor:
        encoder = JointModel(self.embed_len,
                             self.context_embed_len,
                             self.word_embed_len,
                             self.lstm_size,
                             self.num_lstm_layers,
                             self.dropout_keep_prob,
                             self.entity_embeds,
                             pad_vector)
        if not self.load_model:
          log.status('Training')
          trainer = self._get_trainer(cursor, encoder)
          trainer.train()
        else:
          encoder.load_state_dict(torch.load('./model'))
          encoder = nn.DataParallel(encoder)
          encoder = encoder.to(self.device)
        log.status('Testing')
        tester = self._get_tester(cursor, encoder)
        if not self.load_model:
          torch.save(encoder.state_dict(), './model')
        log.report(tester.test())
    finally:
      db_connection.close()
