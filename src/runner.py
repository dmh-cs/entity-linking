from typing import Optional
import math
import pickle

from experiment import Experiment
from pyrsistent import m
from torch.utils.data.sampler import BatchSampler, RandomSampler
import pydash as _
import torch
import torch.nn as nn

from data_fetchers import get_connection, get_embedding_lookup
from joint_model import JointModel
from mention_context_batch_sampler import MentionContextBatchSampler
from mention_context_dataset import MentionContextDataset
from simple_mention_context_dataset_by_entity_ids import SimpleMentionContextDatasetByEntityIds
from tester import Tester
from trainer import Trainer

def load_entity_candidates_and_label_lookup(path, train_size):
  with open(path, 'rb') as lookup_file:
    data = pickle.load(lookup_file)
    assert data['train_size'] == train_size, 'The prior at path ' + path + ' uses train size of ' + \
      str(data['train_size']) + '. Please run `create_candidate_and_entity_lookups.py` with a train size of ' +\
      str(train_size)
    return data['lookups']

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

default_paths = m(lookups='../entity-linking-preprocessing/lookups.pkl',
                  page_id_order='../entity-linking-preprocessing/page_id_order.pkl')
default_train_params = m(batch_size=100,
                         debug=False,
                         use_simple_dataloader=False,
                         num_epochs=1,
                         train_size=0.8,
                         dropout_keep_prob=0.4)
default_model_params = m(use_adaptive_softmax=False,
                         embed_len=100,
                         word_embed_len=100,
                         num_candidates=30,
                         word_embedding_set='glove',
                         local_encoder_lstm_size=100,
                         document_encoder_lstm_size=100,
                         num_lstm_layers=2,
                         ablation=['prior', 'local_context', 'document_context'])
default_run_params = m(load_model=False,
                       comments='')
default_params = default_train_params.update(default_run_params).update(default_model_params)

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
    self.paths = self.paths.set('word_embedding',
                                self._get_word_embedding_path())
    self.page_id_order: Optional[list] = None
    self.num_train_pages: Optional[int] = None
    self.page_id_order_train: Optional[list] = None
    self.page_id_order_test: Optional[list] = None
    self.entity_ids_for_simple_dataset: Optional[list] = None
    self.entity_embeds: Optional[nn.Embedding] = None
    if not self.train_params.use_simple_dataloader and hasattr(self.model_params, 'num_entities'):
      raise NotImplementedError('Can only restrict num of entities when using simple dataloader')

  def _get_word_embedding_path(self):
    if self.model_params.word_embedding_set.lower() == 'glove' and self.model_params.word_embed_len == 100:
      return'./glove.6B.100d.txt'
    elif self.model_params.word_embedding_set.lower() == 'glove' and self.model_params.word_embed_len == 300:
      return './glove.840B.300d.txt'
    else:
      raise NotImplementedError('Only loading from glove 100d or 300d is currently supported')

  def load_caches(self):
    if not hasattr(self.model_params, 'num_entities'):
      self.log.status('Getting number of entities')
      self.model_params = self.model_params.set('num_entities',
                                                get_num_entities())
    self.log.status('Loading entity candidates lookup')
    lookups = load_entity_candidates_and_label_lookup(self.paths.lookups, self.train_params.train_size)
    self.log.status('Loading word embedding lookup')
    self.lookups = self.lookups.update({'entity_candidates_prior': lookups['entity_candidates_prior'],
                                        'entity_labels': lookups['entity_labels'],
                                        'embedding': get_embedding_lookup(self.paths.word_embedding,
                                                                          device=self.device)})
    self.log.status('Getting page id order')
    self.page_id_order = load_page_id_order(self.paths.page_id_order)
    self.num_train_pages = int(len(self.page_id_order) * self.train_params.train_size)
    self.page_id_order_train = self.page_id_order[:self.num_train_pages]
    self.page_id_order_test = self.page_id_order[self.num_train_pages:]
    if self.train_params.use_simple_dataloader:
      self.entity_ids_for_simple_dataset = list(sorted(self.lookups.entity_labels.keys()))[:self.model_params.num_entities]

  def init_entity_embeds(self):
    entity_embed_weights = nn.Parameter(torch.Tensor(self.model_params.num_entities,
                                                     self.model_params.embed_len))
    entity_embed_weights.data.normal_(0, 1.0/math.sqrt(self.model_params.embed_len))
    self.entity_embeds = nn.Embedding(self.model_params.num_entities,
                                      self.model_params.embed_len,
                                      _weight=entity_embed_weights)

  def _get_simple_dataset(self, cursor, is_test):
    return SimpleMentionContextDatasetByEntityIds(cursor,
                                                  self.lookups.entity_candidates_prior,
                                                  self.lookups.entity_labels,
                                                  self.lookups.embedding,
                                                  self.model_params.num_candidates,
                                                  self.entity_ids_for_simple_dataset,
                                                  is_test)

  def _get_dataset(self, cursor, is_test):
    page_ids = self.page_id_order_test if is_test else self.page_id_order_train
    return MentionContextDataset(cursor,
                                 page_ids,
                                 self.lookups.entity_candidates_prior,
                                 self.lookups.entity_labels,
                                 self.lookups.embedding,
                                 self.train_params.batch_size,
                                 self.model_params.num_entities,
                                 self.model_params.num_candidates)

  def _get_sampler(self, cursor, is_test):
    page_ids = self.page_id_order_test if is_test else self.page_id_order_train
    return MentionContextBatchSampler(cursor,
                                      page_ids,
                                      self.train_params.batch_size)

  def _get_trainer(self, cursor, model):
    if self.train_params.use_simple_dataloader:
      train_dataset = self._get_simple_dataset(cursor, is_test=False)
      batch_sampler = BatchSampler(RandomSampler(train_dataset),
                                   self.train_params.batch_size,
                                   True)
    else:
      train_dataset = self._get_dataset(cursor, is_test=False)
      batch_sampler = self._get_sampler(cursor, is_test=False)
    return Trainer(device=self.device,
                   embedding_lookup=self.lookups.embedding,
                   model=model,
                   dataset=train_dataset,
                   batch_sampler=batch_sampler,
                   num_epochs=self.train_params.num_epochs,
                   experiment=self.experiment)

  def _get_tester(self, cursor, model):
    if self.train_params.use_simple_dataloader:
      test_dataset = self._get_simple_dataset(cursor, is_test=True)
      batch_sampler = BatchSampler(RandomSampler(test_dataset),
                                   self.train_params.batch_size,
                                   True)
    else:
      test_dataset = self._get_dataset(cursor, is_test=True)
      batch_sampler = self._get_sampler(cursor, is_test=True)
    return Tester(dataset=test_dataset,
                  batch_sampler=batch_sampler,
                  model=model.mention_context_encoder,
                  entity_embeds=self.entity_embeds,
                  embedding_lookup=self.lookups.embedding,
                  device=self.device,
                  experiment=self.experiment,
                  ablation=self.model_params.ablation)

  def run(self):
    self.load_caches()
    pad_vector = self.lookups.embedding['<PAD>']
    self.init_entity_embeds()
    try:
      db_connection = get_connection()
      with db_connection.cursor() as cursor:
        encoder = JointModel(self.model_params.embed_len,
                             self.model_params.context_embed_len,
                             self.model_params.word_embed_len,
                             self.model_params.local_encoder_lstm_size,
                             self.model_params.document_encoder_lstm_size,
                             self.model_params.num_lstm_layers,
                             self.model_params.use_adaptive_softmax,
                             self.train_params.dropout_keep_prob,
                             self.entity_embeds,
                             pad_vector)
        if not self.run_params.load_model:
          with self.experiment.train(['mention_context_error', 'document_context_error', 'loss']):
            self.log.status('Training')
            trainer = self._get_trainer(cursor, encoder)
            trainer.train()
            torch.save(encoder.state_dict(), './' + self.experiment.model_name)
        else:
          encoder.load_state_dict(torch.load('./' + self.experiment.model_name))
          encoder = nn.DataParallel(encoder)
          encoder = encoder.to(self.device).module
        with self.experiment.test(['accuracy', 'TP', 'num_samples']):
          self.log.status('Testing')
          tester = self._get_tester(cursor, encoder)
          tester.test()
    finally:
      db_connection.close()
