import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, RandomSampler
from trainer import Trainer
from tester import Tester
from joint_model import JointModel
from data_fetchers import get_connection, get_entity_lookup, get_embedding_lookup
from mention_context_dataset import MentionContextDataset
from simple_mention_context_dataset_by_entity_ids import SimpleMentionContextDatasetByEntityIds
from simple_mention_context_dataset import SimpleMentionContextDataset
from mention_context_batch_sampler import MentionContextBatchSampler
import math
import pickle
from random import shuffle
import pydash as _

DEBUG = True
USE_SIMPLE = False

def load_entity_candidates_and_label_lookup(path='../entity-linking-preprocessing/lookups.pkl'):
  with open(path, 'rb') as lookup_file:
    return pickle.load(lookup_file)

def get_num_entities(cursor):
  cursor.execute('select count(*) from entities')
  return cursor.fetchone()['count(*)']

def main(device):
  load_dotenv(dotenv_path='.env')
  LOOKUPS_PATH = os.getenv("LOOKUPS_PATH")
  try:
    num_epochs = 1
    if DEBUG:
      max_num_mentions = 100000
      batch_size = 50
    else:
      max_num_mentions = 10000000
      batch_size = 1000
    num_candidates = 10
    db_connection = get_connection()
    with db_connection.cursor() as cursor:
      print('Loading entity candidates lookup')
      lookups = load_entity_candidates_and_label_lookup(LOOKUPS_PATH)
      entity_candidates_lookup = lookups['entity_candidates']
      entity_label_lookup = lookups['entity_labels']
      # num_entities = len(entity_label_lookup)
      num_entities = 100
      embed_len = 100
      context_embed_len = 2 * embed_len
      print('Creating word embedding lookup')
      embedding_lookup = get_embedding_lookup('./glove.6B.100d.txt',
                                              device=device)
      pad_vector = embedding_lookup['<PAD>']
      word_embed_len = 100
      print('Getting page id order')
      with open('page_id_order.pkl', 'rb') as f:
        page_id_order = pickle.load(f)
      num_train_pages = int(len(page_id_order) * 0.8)
      page_id_order_train = page_id_order[:num_train_pages]
      page_id_order_test = page_id_order[num_train_pages:]
      if USE_SIMPLE:
        entity_ids = list(sorted(entity_label_lookup.keys()))[:num_entities]
        train_dataset = SimpleMentionContextDatasetByEntityIds(cursor,
                                                               entity_candidates_lookup,
                                                               entity_label_lookup,
                                                               embedding_lookup,
                                                               num_candidates,
                                                               entity_ids,
                                                               True)
        batch_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, True)
      else:
        train_dataset = MentionContextDataset(cursor,
                                              page_id_order_train,
                                              entity_candidates_lookup,
                                              entity_label_lookup,
                                              embedding_lookup,
                                              batch_size,
                                              num_entities,
                                              max_num_mentions,
                                              num_candidates)
        batch_sampler = MentionContextBatchSampler(cursor, page_id_order_train, batch_size, max_num_mentions)
      max_num_entities = get_num_entities(cursor)
      embed_len = 100
      entity_embed_weights = nn.Parameter(torch.Tensor(max_num_entities, embed_len))
      entity_embed_weights.data.normal_(0, 1.0/math.sqrt(embed_len))
      entity_embeds = nn.Embedding(max_num_entities, embed_len, _weight=entity_embed_weights)
      lstm_size = 100
      num_lstm_layers = 2
      dropout_keep_prob = 0.5
      encoder = JointModel(embed_len,
                           context_embed_len,
                           word_embed_len,
                           lstm_size,
                           num_lstm_layers,
                           dropout_keep_prob,
                           entity_embeds,
                           pad_vector)
      print('Training')
      trainer = Trainer(device=device,
                        embedding_lookup=embedding_lookup,
                        model=encoder,
                        dataset=train_dataset,
                        batch_sampler=batch_sampler,
                        num_epochs=num_epochs)
      trainer.train()
      # encoder.load_state_dict(torch.load('./model'))
      if USE_SIMPLE:
        test_dataset = SimpleMentionContextDatasetByEntityIds(cursor,
                                                              entity_candidates_lookup,
                                                              entity_label_lookup,
                                                              embedding_lookup,
                                                              num_candidates,
                                                              entity_ids,
                                                              False)
        batch_sampler = BatchSampler(RandomSampler(test_dataset), batch_size, True)
      else:
        test_dataset = MentionContextDataset(cursor,
                                             page_id_order_test,
                                             entity_candidates_lookup,
                                             entity_label_lookup,
                                             embedding_lookup,
                                             batch_size,
                                             num_entities,
                                             max_num_mentions,
                                             num_candidates)
        batch_sampler = MentionContextBatchSampler(cursor, page_id_order_test, batch_size, max_num_mentions)
      tester = Tester(dataset=test_dataset,
                      batch_sampler=batch_sampler,
                      model=encoder.module.mention_context_encoder,
                      entity_embeds=entity_embeds,
                      embedding_lookup=embedding_lookup,
                      device=device)
      torch.save(encoder.state_dict(), './model')
      print(tester.test())
  finally:
    db_connection.close()

if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)
  except:
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
