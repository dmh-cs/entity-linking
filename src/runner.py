from typing import Optional
import math

from experiment import Experiment
from pyrsistent import m, ny
import pydash as _
import torch
import torch.nn as nn
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

from data_fetchers import get_connection, get_embedding_dict, get_num_entities, load_page_id_order, load_entity_candidate_ids_and_label_lookup, get_entity_text
from default_params import default_train_params, default_model_params, default_run_params, default_paths
from joint_model import JointModel
from logits import Logits
from mention_context_batch_sampler import MentionContextBatchSampler
from mention_context_dataset import MentionContextDataset
from softmax import Softmax
from tester import Tester
from trainer import Trainer
from parsers import parse_for_tokens
from data_transformers import pad_batch_list

from fire_extinguisher import BatchRepeater

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
                                              self.model_params.embed_len)
    if not hasattr(self.model_params, 'adaptive_softmax_cutoffs'):
      self.model_params = self.model_params.set('adaptive_softmax_cutoffs',
                                                [1000, 10000, 100000, 300000, 500000])
    self.paths = self.paths.set('word_embedding',
                                self._get_word_embedding_path())
    self.page_id_order: Optional[list] = None
    self.num_train_pages: Optional[int] = None
    self.page_id_order_train: Optional[list] = None
    self.page_id_order_test: Optional[list] = None
    self.entity_embeds: Optional[nn.Embedding] = None
    self.adaptive_logits = {'desc': None, 'mention': None}
    self.encoder = None

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

  def load_caches(self):
    if not hasattr(self.model_params, 'num_entities'):
      self.log.status('Getting number of entities')
      self.model_params = self.model_params.set('num_entities',
                                                get_num_entities())
    self.log.status('Loading entity candidate_ids lookup')
    lookups = load_entity_candidate_ids_and_label_lookup(self.paths.lookups, self.train_params.train_size)
    self.log.status('Loading word embedding lookup')
    embedding_dict = get_embedding_dict(self.paths.word_embedding,
                                        embedding_dim=self.model_params.word_embed_len)
    token_idx_lookup = dict(zip(embedding_dict.keys(),
                                range(len(embedding_dict))))
    embedding = nn.Embedding.from_pretrained(torch.stack([embedding_dict[token] for token in token_idx_lookup]).to(self.device),
                                             freeze=self.model_params.freeze_word_embeddings)
    self.lookups = self.lookups.update({'entity_candidates_prior': lookups['entity_candidates_prior'],
                                        'entity_labels': lookups['entity_labels'],
                                        'embedding': embedding,
                                        'token_idx_lookup': token_idx_lookup})
    self.log.status('Getting page id order')
    self.page_id_order = load_page_id_order(self.paths.page_id_order)
    self.num_train_pages = int(len(self.page_id_order) * self.train_params.train_size)
    self.page_id_order_train = self.page_id_order[:self.num_train_pages]
    self.page_id_order_test = self.page_id_order[self.num_train_pages:]

  def _get_entity_tokens(self):
    mapper = lambda token: self.lookups.token_idx_lookup[token] if token in self.lookups.token_idx_lookup else self.lookups.token_idx_lookup['<UNK>']
    entity_indexed_tokens = {self.lookups.entity_labels[entity_id]: _.map_(parse_for_tokens(text), mapper)
                             for entity_id, text in get_entity_text().items()
                             if entity_id in self.lookups.entity_labels}
    entity_indexed_tokens_list = [entity_indexed_tokens[i]
                                  if i in entity_indexed_tokens else [1]
                                  for i in range(len(entity_indexed_tokens))]
    return torch.tensor(pad_batch_list(0, entity_indexed_tokens_list),
                        device=self.device)

  def _sum_in_batches(self, by_token):
    results = []
    for chunk in torch.chunk(by_token, 100):
      entity_word_vecs = self.lookups.embedding(chunk)
      results.append(entity_word_vecs.sum(1))
    return torch.cat(results, 0)

  def init_entity_embeds(self):
    if self.model_params.word_embed_len == self.model_params.embed_len:
      entities_by_token = self._get_entity_tokens()
      entity_embed_weights = nn.Parameter(self._sum_in_batches(entities_by_token))
    else:
      print(f'word embed len: {self.model_params.word_embed_len} != entity embed len {self.model_params.embed_len}. Not initializing entity embeds with word embeddings')
      entity_embed_weights = nn.Parameter(torch.Tensor(self.model_params.num_entities,
                                                       self.model_params.embed_len))
      entity_embed_weights.data.normal_(0, 1.0/math.sqrt(self.model_params.embed_len))
    self.entity_embeds = nn.Embedding(self.model_params.num_entities,
                                      self.model_params.embed_len,
                                      _weight=entity_embed_weights).to(self.device)

  def _get_dataset(self, cursor, is_test):
    page_ids = self.page_id_order_test if is_test else self.page_id_order_train
    return MentionContextDataset(cursor,
                                 page_ids,
                                 self.lookups.entity_candidates_prior,
                                 self.lookups.entity_labels,
                                 self.lookups.embedding,
                                 self.lookups.token_idx_lookup,
                                 self.train_params.batch_size,
                                 self.model_params.num_entities,
                                 self.model_params.num_candidates,
                                 cheat=self.run_params.cheat)

  def _get_sampler(self, cursor, is_test, limit=None):
    page_ids = self.page_id_order_test if is_test else self.page_id_order_train
    return MentionContextBatchSampler(cursor,
                                      page_ids,
                                      self.train_params.batch_size,
                                      limit=limit)

  def _calc_logits(self, encoded, candidate_entity_ids):
    desc_embeds, mention_context_embeds = encoded
    if self.model_params.use_adaptive_softmax:
      raise NotImplementedError('No longer supported')
    elif self.model_params.use_ranking_loss:
      raise NotImplementedError('No longer supported')
    else:
      logits = Logits()
      desc_logits = logits(desc_embeds,
                           self.entity_embeds(candidate_entity_ids))
      mention_logits = logits(mention_context_embeds,
                              self.entity_embeds(candidate_entity_ids))
    return desc_logits, mention_logits

  def _calc_loss(self, scores, labels_for_batch):
    desc_score, mention_score = scores
    if self.model_params.use_adaptive_softmax:
      raise NotImplementedError('No longer supported')
    elif self.model_params.use_ranking_loss:
      raise NotImplementedError('No longer supported')
    else:
      criterion = nn.CrossEntropyLoss()
      desc_loss = criterion(desc_score, labels_for_batch)
      mention_loss = criterion(mention_score, labels_for_batch)
    return desc_loss + mention_loss

  def _get_trainer(self, cursor, model):
    return Trainer(device=self.device,
                   embedding=self.lookups.embedding,
                   token_idx_lookup=self.lookups.token_idx_lookup,
                   model=model,
                   get_dataset=lambda: self._get_dataset(cursor, is_test=False),
                   get_batch_sampler=lambda: self._get_sampler(cursor,
                                                               is_test=False,
                                                               limit=self.train_params.dataset_limit),
                   num_epochs=self.train_params.num_epochs,
                   experiment=self.experiment,
                   calc_loss=self._calc_loss,
                   calc_logits=self._calc_logits,
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
                  embedding=self.lookups.embedding,
                  token_idx_lookup=self.lookups.token_idx_lookup,
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
      return AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=1.0).to(self. device)
    calc = get_calc('desc_and_mention')
    return {context: calc for context in ['desc', 'mention']}

  def run(self):
    self.load_caches()
    pad_vector = self.lookups.embedding(torch.tensor([self.lookups.token_idx_lookup['<PAD>']],
                                                     device=self.lookups.embedding.weight.device)).squeeze()
    self.init_entity_embeds()
    try:
      db_connection = get_connection()
      with db_connection.cursor() as cursor:
        entity_ids_by_freq = self._get_entity_ids_by_freq(cursor)
        if self.model_params.use_adaptive_softmax:
          self.lookups = self.lookups.set('entity_labels',
                                          _.from_pairs(zip(entity_ids_by_freq,
                                                           range(len(entity_ids_by_freq)))))
        self.adaptive_logits = self._get_adaptive_calc_logits()
        self.encoder = JointModel(self.model_params.embed_len,
                                  self.model_params.context_embed_len,
                                  self.model_params.word_embed_len,
                                  self.model_params.local_encoder_lstm_size,
                                  self.model_params.document_encoder_lstm_size,
                                  self.model_params.num_lstm_layers,
                                  self.train_params.dropout_drop_prob,
                                  self.entity_embeds,
                                  self.lookups.embedding,
                                  pad_vector,
                                  self.adaptive_logits,
                                  self.model_params.use_deep_network,
                                  self.model_params.use_lstm_local,
                                  self.model_params.num_cnn_local_filters,
                                  self.model_params.use_cnn_local)
        if not self.run_params.load_model:
          with self.experiment.train(['mention_context_error', 'document_context_error', 'loss']):
            self.log.status('Training')
            trainer = self._get_trainer(cursor, self.encoder)
            trainer.train()
            torch.save(self.encoder.state_dict(), './' + self.experiment.model_name)
        else:
          self.encoder.load_state_dict(torch.load('./' + self.experiment.model_name))
          self.encoder = nn.DataParallel(self.encoder)
          self.encoder = self.encoder.to(self.device).module
        with self.experiment.test(['accuracy', 'TP', 'num_samples']):
          self.log.status('Testing')
          tester = self._get_tester(cursor, self.encoder)
          tester.test()
    finally:
      db_connection.close()
