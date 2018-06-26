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

def load_entity_candidates_lookup(path='../entity-linking-preprocessing/candidate_lookup.pkl'):
  with open(path, 'rb') as lookup_file:
    return pickle.load(lookup_file)

def get_page_id_order(cursor):
  cursor.execute('select id from pages')
  page_ids = []
  while True:
    results = cursor.fetchmany(10000)
    if results is None: break
    page_ids.append([row['id'] for row in results])
  return shuffle(page_ids)

def main():
  try:
    batch_size = 1000
    num_epochs = 1000
    max_num_mentions = 100000
    db_connection = get_connection()
    with db_connection.cursor() as cursor:
      entity_candidates_lookup = load_entity_candidates_lookup()
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
                                      batch_size,
                                      max_num_mentions)
      batch_sampler = MentionContextBatchSampler(cursor, page_id_order, batch_size)
      dataloader = DataLoader(dataset, batch_size=batch_size, batch_sampler=batch_sampler)
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


if __name__ == "__main__": main()
