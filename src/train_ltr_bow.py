import os
from random import shuffle

import pickle
import pymysql.cursors
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from progressbar import progressbar
import json

from utils import tensors_to_device, to_idx
from ltr_bow import LtRBoW
from simple_mention_dataset import SimpleMentionDataset, collate_simple_mention_pointwise, collate_simple_mention_pairwise
from losses import hinge_loss
from simple_conll_dataset import SimpleCoNLLDataset, collate_simple_mention_ranker
from samplers import SubsetSequentialSampler
from early_stopping_helpers import eval_model

from rabbit_ml import get_cli_args

from args_config import args

def main():
  p = get_cli_args(args)
  with open('./tokens.pkl', 'rb') as fh: token_idx_lookup = pickle.load(fh)
  with open('./glove_token_idx_lookup.pkl', 'rb') as fh: full_token_idx_lookup = pickle.load(fh)
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
    conll_path = 'custom.tsv' if p.run.use_custom else './AIDA-YAGO2-dataset.tsv'
    test_dataset = SimpleCoNLLDataset(cursor,
                                      token_idx_lookup,
                                      full_token_idx_lookup,
                                      conll_path,
                                      p.run.lookups_path,
                                      p.run.idf_path,
                                      p.train.train_size,
                                      p.run.txt_dataset_path)
    try:
      with open('./val_test_indices.json', 'r') as fh:
        val_indices, test_indices = json.load(fh)
    except FileNotFoundError:
      with open('./val_test_indices.json', 'w') as fh:
        permutation = list(range(len(test_dataset))); shuffle(permutation)
        split_idx = int(len(test_dataset) * 0.25)
        val_indices, test_indices = permutation[:split_idx], permutation[split_idx:]
        json.dump((val_indices, test_indices), fh)
    test_dataloader = DataLoader(test_dataset,
                                 batch_sampler=BatchSampler(SubsetSequentialSampler(val_indices),
                                                            p.run.batch_size,
                                                            False),
                                 collate_fn=collate_simple_mention_ranker)
    collate_fn = collate_simple_mention_pairwise if p.train.use_pairwise else collate_simple_mention_pointwise
    if p.train.train_on_conll:
      conll_path = 'custom.tsv' if p.run.use_custom else './AIDA-YAGO2-dataset.tsv'
      dataset = SimpleCoNLLDataset(cursor,
                                   token_idx_lookup,
                                   full_token_idx_lookup,
                                   conll_path,
                                   p.run.lookups_path,
                                   p.run.idf_path,
                                   p.train.train_size,
                                   txt_dataset_path=p.run.txt_dataset_path)
    else:
      dataset = SimpleMentionDataset(cursor,
                                     token_idx_lookup,
                                     full_token_idx_lookup,
                                     page_ids,
                                     p.run.lookups_path,
                                     p.run.idf_path,
                                     p.train.train_size,
                                     txt_dataset_path=p.run.txt_dataset_path)
    sampler = SequentialSampler if p.train.use_sequential_sampler else RandomSampler
    dataloader = DataLoader(dataset,
                            batch_sampler=BatchSampler(sampler(dataset), p.train.batch_size, False),
                            collate_fn=collate_fn)
    model = LtRBoW(p.model.hidden_sizes, dropout_keep_prob=p.train.dropout_keep_prob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    calc_loss = nn.MarginRankingLoss(p.train.margin) if p.train.use_hinge else nn.BCEWithLogitsLoss()
    with open('./losses_ltr' + ','.join(str(sz) for sz in p.model.hidden_sizes), 'w') as fh:
      for epoch_num in range(p.train.num_epochs):
        print(eval_model(test_dataloader, model, device))
        for batch_num, batch in progressbar(enumerate(dataloader)):
          model.train()
          optimizer.zero_grad()
          if p.train.use_pairwise:
            features, labels = batch
            features = [elem.to(device) for elem in features]
            labels = labels.to(device)
            target_features, candidate_features = features
            target_scores = model(target_features)
            candidate_scores = model(candidate_features)
            scores = candidate_scores - target_scores
          else:
            batch = [elem.to(device) for elem in batch]
            features, labels = batch
            scores = model(features)
          if p.train.use_hinge:
            loss = calc_loss(target_scores, candidate_scores, torch.ones_like(labels))
          else:
            loss = calc_loss(scores, labels)
          fh.write('{}\n'.format(loss.item()))
          fh.flush()
          loss.backward()
          optimizer.step()
          train_str = 'pairwise' if p.train.use_pairwise else ''
          loss_str = 'hinge_{}'.format(p.train.margin) if p.train.use_hinge else ''
          torch.save(model.state_dict(), './ltr_model_' + ','.join(str(sz) for sz in p.model.hidden_sizes) + train_str + '_' + loss_str)



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
