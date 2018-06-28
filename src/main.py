import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer import Trainer
from mention_context_encoder_model import MentionContextEncoder
from data_fetchers import get_connection, get_entity_lookup, get_embedding_lookup
from mention_context_dataset import MentionContextDataset
from mention_context_batch_sampler import MentionContextBatchSampler
import math
import pickle
from random import shuffle
import pydash as _

def load_entity_candidates_and_label_lookup(path='../entity-linking-preprocessing/lookups.pkl'):
  with open(path, 'rb') as lookup_file:
    return pickle.load(lookup_file)

def get_page_id_order(cursor):
  cursor.execute('select id from pages')
  page_ids = []
  while True:
    results = cursor.fetchmany(10000)
    if _.is_empty(results): break
    page_ids.extend([row['id'] for row in results])
  shuffle(page_ids)
  return page_ids

def get_num_entities(cursor):
  cursor.execute('select count(*) from entities')
  return cursor.fetchone()['count(*)']

def collate(batch):
  max_num_candidates = max(map(len, [sample['candidates'] for sample in batch]))
  return {'sentence_splits': [sample['sentence_splits'] for sample in batch],
          'label': torch.tensor([sample['label'] for sample in batch]),
          'document_mention_indices': [sample['document_mention_indices'] for sample in batch],
          'candidates': [sample['candidates'] for sample in batch]}

def main():
  try:
    batch_size = 10
    num_epochs = 1000
    max_num_mentions = 10000000
    db_connection = get_connection()
    with db_connection.cursor() as cursor:
      print('Loading entity candidates lookup')
      lookups = load_entity_candidates_and_label_lookup()
      entity_candidates_lookup = lookups['entity_candidates']
      entity_label_lookup = lookups['entity_labels']
      embed_len = 100
      context_embed_len = 2 * embed_len
      print('Creating word embedding lookup')
      embedding_lookup = get_embedding_lookup('./glove.6B.100d.txt')
      word_embed_len = 100
      print('Getting page id order')
      page_id_order = get_page_id_order(cursor)
      dataset = MentionContextDataset(cursor,
                                      page_id_order,
                                      entity_candidates_lookup,
                                      entity_label_lookup,
                                      batch_size,
                                      max_num_mentions)
      batch_sampler = MentionContextBatchSampler(cursor, page_id_order, batch_size)
      dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate)
      num_entities = get_num_entities(cursor)
      embed_len = 100
      entity_embed_weights = nn.Parameter(torch.Tensor(num_entities, embed_len))
      entity_embed_weights.data.normal_(0, 1.0/math.sqrt(embed_len))
      entity_embeds = nn.Embedding(num_entities, embed_len, _weight=entity_embed_weights)
      lstm_size = 100
      num_lstm_layers = 2
      dropout_keep_prob = 0.5
      encoder = MentionContextEncoder(embed_len,
                                      context_embed_len,
                                      word_embed_len,
                                      max_num_mentions,
                                      lstm_size,
                                      num_lstm_layers,
                                      dropout_keep_prob,
                                      entity_embeds)
      print('Training')
      trainer = Trainer(embedding_lookup=embedding_lookup,
                        model=encoder,
                        dataset=dataloader,
                        num_epochs=num_epochs)
      trainer.train(batch_size)
  finally:
    db_connection.close()

if __name__ == "__main__":
  import pdb
  import traceback
  import sys

  try:
    main()
  except:
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
  # main()
