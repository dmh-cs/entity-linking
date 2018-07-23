import os
import random

from dotenv import load_dotenv
import numpy as np
import pydash as _
import pymysql.cursors
import torch

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

def get_embedding_lookup(path, embedding_dim=100, device=None):
  if device is None: raise ValueError('Specify a device')
  lookup = {'<PAD>': torch.zeros(size=(embedding_dim,), dtype=torch.float32, device=device),
            '<UNK>': torch.randn(size=(embedding_dim,), dtype=torch.float32, device=device),
            '<MENTION_START_HERE>': torch.randn(size=(embedding_dim,), dtype=torch.float32, device=device),
            '<MENTION_END_HERE>': torch.randn(size=(embedding_dim,), dtype=torch.float32, device=device)}
  with open(path) as f:
    while True:
      line = f.readline()
      if line and len(line) > 0:
        split_line = line.strip().split(' ')
        lookup[split_line[0]] = torch.tensor(np.array(split_line[1:], dtype=np.float32),
                                             dtype=torch.float32,
                                             device=device)
      else:
        break
  return lookup

def get_random_indexes(max_value, exclude, num_to_generate):
  result = []
  while len(result) < num_to_generate:
    val = random.randint(0, max_value - 1)
    while val in exclude:
      val = random.randint(0, max_value - 1)
    result.append(val)
  return result

def get_candidates(entity_candidates_lookup,
                   num_entities,
                   num_candidates,
                   mention,
                   label):
  base_candidates = entity_candidates_lookup[mention]
  if len(base_candidates) < num_candidates:
    num_candidates_to_generate = num_candidates - len(base_candidates)
    random_candidates = get_random_indexes(num_entities,
                                           base_candidates.tolist(),
                                           num_candidates_to_generate)
    candidates = torch.cat((base_candidates,
                           torch.tensor(random_candidates)), 0)
  else:
    label_index = int((base_candidates == label).nonzero().squeeze())
    random_candidates = get_random_indexes(len(base_candidates),
                                           [label_index],
                                           num_candidates - 1)
    indexes_to_keep = random_candidates + [label_index]
    candidates = base_candidates[indexes_to_keep]
  order = list(range(num_candidates))
  random.shuffle(order)
  return candidates[order]
