import os

import pickle
import pymysql.cursors
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from progressbar import progressbar

from utils import tensors_to_device, to_idx
from ltr_bow import LtRBoW
from simple_mention_dataset import SimpleMentionDataset, collate_simple_mention_pointwise, collate_simple_mention_pairwise
from losses import hinge_loss

from rabbit_ml import get_cli_args, list_arg, optional_arg

args =  [{'name': 'num_epochs',       'for': 'train_params', 'type': int, 'default': 5},
         {'name': 'batch_size',       'for': 'train_params', 'type': int, 'default': 512},
         {'name': 'num_pages_to_use', 'for': 'train_params', 'type': int, 'default': 10000},
         {'name': 'use_pairwise',     'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_hinge',        'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'page_id_order_path', 'for': 'train_params', 'type': str, 'default': '../wp-preprocessing-el/page_id_order.pkl_local'},
         {'name': 'lookups_path',     'for': 'run_params', 'type': str, 'default': '../wp-preprocessing-el/lookups.pkl_local'},
         {'name': 'idf_path',         'for': 'run_params', 'type': str, 'default': './wiki_idf_stem.json'},
         {'name': 'hidden_sizes',     'for': 'model_params', 'type': list_arg(str), 'default': [100]},
         {'name': 'env_path',         'for': 'run_params', 'type': str, 'default': '.env'},
         {'name': 'train_size',       'for': 'train_params', 'type': float, 'default': 1.0}]

def main():
  p = get_cli_args(args)
  model = LtRBoW(p.model.hidden_sizes)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  optimizer = optim.Adam(model.parameters())
  load_dotenv(dotenv_path=p.run.env_path)
  EL_DATABASE_NAME = os.getenv("DBNAME")
  DATABASE_USER = os.getenv("DBUSER")
  DATABASE_PASSWORD = os.getenv("DBPASS")
  DATABASE_HOST = os.getenv("DBHOST")
  with open(p.train.page_id_order_path, 'rb') as fh:
    page_id_order = pickle.load(fh)
  page_ids = page_id_order[:p.train.num_pages_to_use]
  connection = pymysql.connect(host=DATABASE_HOST,
                               user=DATABASE_USER,
                               password=DATABASE_PASSWORD,
                               db=EL_DATABASE_NAME,
                               charset='utf8mb4',
                               use_unicode=True,
                               cursorclass=pymysql.cursors.DictCursor)
  with connection.cursor() as cursor:
    cursor.execute("SET NAMES utf8mb4;")
    cursor.execute("SET CHARACTER SET utf8mb4;")
    cursor.execute("SET character_set_connection=utf8mb4;")
    collate_fn = collate_simple_mention_pairwise if p.train.use_pairwise else collate_simple_mention_pointwise
    dataset = SimpleMentionDataset(cursor, page_ids, p.run.lookups_path, p.run.idf_path, p.train.train_size)
    dataloader = DataLoader(dataset,
                            batch_sampler=BatchSampler(RandomSampler(dataset), p.train.batch_size, False),
                            collate_fn=collate_fn)
    calc_loss = hinge_loss if p.train.use_hinge else nn.BCEWithLogitsLoss()
    with open('./losses_ltr', 'w') as fh:
      for epoch_num in range(p.train.num_epochs):
        for batch_num, batch in progressbar(enumerate(dataloader)):
          model.train()
          optimizer.zero_grad()
          batch = [elem.to(device) for elem in batch]
          features, labels = batch
          scores = model(features)
          loss = calc_loss(scores, labels)
          fh.write('{}\n'.format(loss.item()))
          fh.flush()
          loss.backward()
          optimizer.step()
    torch.save(model.state_dict(), './ltr_model_' + ','.join(p.model.hidden_sizes))



if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
