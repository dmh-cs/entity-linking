import os

import pickle
import pymysql.cursors
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from progressbar import progressbar

from utils import tensors_to_device, to_idx
from ltr_bow import LtRBoW
from simple_mention_dataset import SimpleMentionDataset, collate_simple_mention_pointwise, collate_simple_mention_pairwise
from losses import hinge_loss
from simple_conll_dataset import SimpleCoNLLDataset

from rabbit_ml import get_cli_args

from args_config import args

def main():
  p = get_cli_args(args)
  model = LtRBoW(p.model.hidden_sizes, dropout_keep_prob=p.train.dropout_keep_prob)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  optimizer = optim.Adam(model.parameters())
  with open('./tokens.pkl', 'rb') as fh: token_idx_lookup = pickle.load(fh)
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
    if p.train.train_on_conll:
      conll_path = 'custom.tsv' if p.run.use_custom else './AIDA-YAGO2-dataset.tsv'
      dataset = SimpleCoNLLDataset(cursor,
                                   token_idx_lookup,
                                   conll_path,
                                   p.run.lookups_path,
                                   p.run.idf_path,
                                   p.train.train_size)
    else:
      dataset = SimpleMentionDataset(cursor,
                                     token_idx_lookup,
                                     page_ids,
                                     p.run.lookups_path,
                                     p.run.idf_path,
                                     p.train.train_size)
    sampler = SequentialSampler if p.train.use_sequential_sampler else RandomSampler
    dataloader = DataLoader(dataset,
                            batch_sampler=BatchSampler(sampler(dataset), p.train.batch_size, False),
                            collate_fn=collate_fn)
    calc_loss = hinge_loss if p.train.use_hinge else nn.BCEWithLogitsLoss()
    with open('./losses_ltr' + ','.join(str(sz) for sz in p.model.hidden_sizes), 'w') as fh:
      for epoch_num in range(p.train.num_epochs):
        for batch_num, batch in progressbar(enumerate(dataloader)):
          model.train()
          optimizer.zero_grad()
          batch = [elem.to(device) for elem in batch]
          features, labels = batch
          if p.train.use_pairwise:
            target_features, candidate_features = features
            target_scores = model(target_features)
            candidate_scores = model(candidate_features)
            scores = target_scores - candidate_scores
          else:
            scores = model(features)
          loss = calc_loss(scores, labels)
          fh.write('{}\n'.format(loss.item()))
          fh.flush()
          loss.backward()
          optimizer.step()
    train_str = 'pairwise' if p.train.use_pairwise else ''
    torch.save(model.state_dict(), './ltr_model_' + ','.join(str(sz) for sz in p.model.hidden_sizes) + train_str)



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
