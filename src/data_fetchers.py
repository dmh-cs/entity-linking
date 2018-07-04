import os
import pymysql.cursors
import numpy as np
import torch
from dotenv import load_dotenv
from pathlib import Path
import pydash as _
from functools import reduce

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

def get_entity_lookup():
  try:
    connection = get_connection()
    with get_cursor(connection) as cursor:
      cursor.execute('select * from entities')
      entities = cursor.fetchall()
      return reduce(lambda lookup, entity: _.assign(lookup, {entity['text']: entity['id']}),
                    entities,
                    {})
  finally:
    connection.close()

def get_embedding_lookup(path, embedding_dim=100):
  lookup = {'<PAD>': torch.rand(size=(embedding_dim,), dtype=torch.float32),
            '<UNK>': torch.rand(size=(embedding_dim,), dtype=torch.float32),
            '<MENTION_START_HERE>': torch.rand(size=(embedding_dim,), dtype=torch.float32),
            '<MENTION_END_HERE>': torch.rand(size=(embedding_dim,), dtype=torch.float32)}
  with open(path) as f:
    while True:
      line = f.readline()
      if line and len(line) > 0:
        split_line = line.strip().split(' ')
        lookup[split_line[0]] = torch.tensor(np.array(split_line[1:], dtype=np.float32), dtype=torch.float32)
      else:
        break
  return lookup
