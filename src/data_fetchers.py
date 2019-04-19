import os
import random
import pickle

from dotenv import load_dotenv
import numpy as np
import pydash as _
import pymysql.cursors
import torch
import torch.nn as nn
from progressbar import progressbar

import utils as u

def get_connection():
  load_dotenv(dotenv_path='.env')
  DATABASE_NAME = os.getenv("DBNAME")
  DATABASE_USER = os.getenv("DBUSER")
  DATABASE_PASSWORD = os.getenv("DBPASS")
  DATABASE_HOST = os.getenv("DBHOST")
  connection = pymysql.connect(host=DATABASE_HOST,
                               user=DATABASE_USER,
                               password=DATABASE_PASSWORD,
                               db=DATABASE_NAME,
                               charset='utf8mb4',
                               use_unicode=True,
                               cursorclass=pymysql.cursors.DictCursor)
  return connection

def get_cursor(db_connection):
  cursor = db_connection.cursor()
  cursor.execute("SET NAMES utf8mb4;")
  cursor.execute("SET CHARACTER SET utf8mb4;")
  cursor.execute("SET character_set_connection=utf8mb4;")
  return cursor

def get_train_validation_test_cursors(db_connection):
  cursors = {}
  for cursor_name in ['train', 'validation', 'test']:
    cursor = db_connection.cursor()
    cursor.execute("SET NAMES utf8mb4;")
    cursor.execute("SET CHARACTER SET utf8mb4;")
    cursor.execute("SET character_set_connection=utf8mb4;")
    cursors[cursor_name] = cursor
  return cursors

def close_cursors(cursors):
  for cursor_name, cursor in cursors.items():
    cursor.close()

def _get_data_fetcher(num_items_per_dataset):
  splits = {}
  split_index = 0
  for dataset_name, num_items in num_items_per_dataset.items():
    splits[dataset_name] = [split_index, split_index + num_items]
    split_index += num_items
  def __data_fetcher(cursor, dataset_name):
    cursor.execute('select * from pages where is_seed_page = 1 limit %s, %s', splits[dataset_name])
    return u.build_cursor_generator(cursor)
  return __data_fetcher

def get_raw_datasets(cursors, num_items):
  return _.map_values(cursors, _get_data_fetcher(num_items))

def get_embedding_dict(path, embedding_dim=100):
  lookup = {'<PAD>': torch.zeros(size=(embedding_dim,), dtype=torch.float32),
            '<UNK>': torch.randn(size=(embedding_dim,), dtype=torch.float32),
            '<MENTION_START_HERE>': torch.randn(size=(embedding_dim,), dtype=torch.float32),
            '<MENTION_END_HERE>': torch.randn(size=(embedding_dim,), dtype=torch.float32)}
  with open(path) as f:
    for line in progressbar(f):
      split_line = line.rstrip().split(' ')
      lookup[split_line[0]] = torch.tensor(np.array(split_line[1:], dtype=np.float32),
                                           dtype=torch.float32)
  return lookup

def get_random_indexes(max_value, exclude, num_to_generate):
  if max_value < num_to_generate:
    raise ValueError
  result = []
  while len(result) < num_to_generate:
    val = random.randint(0, max_value - 1)
    while val in exclude or val in result:
      val = random.randint(0, max_value - 1)
    result.append(val)
  return result

def get_candidate_ids(entity_candidates_prior,
                      num_entities,
                      num_candidates,
                      mention,
                      label,
                      cheat=False):
  if cheat:
    if entity_candidates_prior.get(mention) is None:
      base_candidate_ids = torch.tensor([label], dtype=torch.long)
    else:
      ids = list(entity_candidates_prior[mention].keys())
      base_candidate_ids = torch.tensor(ids + [label] if label not in ids else ids,
                                        dtype=torch.long)
  else:
    if entity_candidates_prior.get(mention) is None:
      base_candidate_ids = torch.tensor([], dtype=torch.long)
    else:
      ids = list(entity_candidates_prior[mention].keys())
      base_candidate_ids = torch.tensor(ids, dtype=torch.long)
  if len(base_candidate_ids) < num_candidates:
    num_candidates_to_generate = num_candidates - len(base_candidate_ids)
    random_candidate_ids = get_random_indexes(num_entities,
                                              base_candidate_ids.tolist(),
                                              num_candidates_to_generate)
    candidate_ids = torch.cat((base_candidate_ids,
                               torch.tensor(random_candidate_ids)), 0)
  else:
    label_in_candidates_indexes = (base_candidate_ids == label).nonzero()
    if len(label_in_candidates_indexes) == 0:
      indexes_to_keep = get_random_indexes(len(base_candidate_ids),
                                           [],
                                           num_candidates)
    else:
      label_index = int(label_in_candidates_indexes.squeeze())
      random_candidate_ids = get_random_indexes(len(base_candidate_ids),
                                                [label_index],
                                                num_candidates - 1)
      indexes_to_keep = random_candidate_ids + [label_index]
    candidate_ids = base_candidate_ids[indexes_to_keep]
  order = list(range(num_candidates))
  random.shuffle(order)
  return candidate_ids[order]

def load_entity_candidate_ids_and_label_lookup(path, train_size):
  with open(path, 'rb') as lookup_file:
    data = pickle.load(lookup_file)
    assert data['train_size'] == train_size, 'The prior at path ' + path + ' uses train size of ' + \
      str(data['train_size']) + \
      '. Please run `create_candidate_and_entity_lookups.py` with a train size of ' +\
      str(train_size)
    return data['lookups']

def load_page_id_order(path):
  with open(path, 'rb') as f:
    return pickle.load(f)

def get_num_entities():
  try:
    db_connection = get_connection()
    with db_connection.cursor() as cursor:
      cursor.execute('select count(*) from (select entity_id from entity_mentions group by `entity_id`) ct')
      return cursor.fetchone()['count(*)']
  finally:
    db_connection.close()

def get_entity_text():
  try:
    db_connection = get_connection()
    with db_connection.cursor() as cursor:
      cursor.execute('select id, text from entities')
      return {row['id']: row['text'] for row in cursor.fetchall()}
  finally:
    db_connection.close()

def get_p_prior(entity_candidates_prior, mention, candidate_ids):
  if mention not in entity_candidates_prior:
    return torch.zeros(len(candidate_ids))
  entity_counts = entity_candidates_prior[mention]
  candidate_counts = [entity_counts[entity] if entity in entity_counts else 0 for entity in candidate_ids.tolist()]
  return torch.tensor(candidate_counts, dtype=torch.float) / sum(candidate_counts)

def get_candidate_strs(cursor, candidate_ids):
  cursor.execute('select id, text from entities where id in (' + str(candidate_ids)[1:-1] + ')')
  lookup = {row['id']: row['text'] for row in cursor.fetchall()}
  return [lookup[cand_id] for cand_id in candidate_ids]
