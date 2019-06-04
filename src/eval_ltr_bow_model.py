from collections import Counter
from itertools import count
import json
import wikipedia
from progressbar import progressbar
import sys
import requests
import string
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.data import DataLoader
import torch

import numpy as np
from nltk.stem.snowball import SnowballStemmer

from simple_conll_dataset import SimpleCoNLLDataset, collate_simple_mention_ranker
from parsers import parse_text_for_tokens
from data_fetchers import get_connection
from fixed_weights_model import FixedWeights
from ltr_bow import LtRBoW

from rabbit_ml import get_cli_args, list_arg, optional_arg

args =  [{'name': 'batch_size',       'for': 'train_params', 'type': int, 'default': 512},
         {'name': 'num_pages_to_use', 'for': 'train_params', 'type': int, 'default': 10000},
         {'name': 'page_id_order_path', ' for': 'train_params', 'type': str, 'default': '../wp-entity-preprocessing/page_id_order.pkl_local'},
         {'name': 'lookups_path',     'for': 'run_params', 'type': str, 'default': '../wp-preprocessing-el/lookups.pkl_local'},
         {'name': 'idf_path',         'for': 'run_params', 'type': str, 'default': './wiki_idf_stem.json'},
         {'name': 'env_path',         'for': 'run_params', 'type': str, 'default': '.env'},
         {'name': 'use_custom',       'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'just_tfidf',       'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'train_size',       'for': 'train_params', 'type': float, 'default': 1.0}]


def load_model(model_params):
  if model_params.just_tfidf:
    return FixedWeights([1, 0, 0, 0, 0])
  else:
    model = LtRBoW(model_params.hidden_sizes)
    path = './ltr_model_' + ','.join(model_params.hidden_sizes)
    model.load_state_dict(torch.load(path))
    return model

def main():
  p = get_cli_args(args)
  conll_path = 'custom.tsv' if p.run.use_custom else './AIDA-YAGO2-dataset.tsv'
  num_correct = 0
  missed_idxs = []
  guessed_when_missed = []
  db_connection = get_connection(p.run.env_path)
  model = load_model(p.model)
  with db_connection.cursor() as cursor:
    dataset = SimpleCoNLLDataset(cursor, conll_path, p.lookups_path, p.run.idf_path, p.train.train_size)
    conll_test_set = DataLoader(dataset,
                                batch_sampler=BatchSampler(SequentialSampler(dataset),
                                                           p.run.batch_size,
                                                           False),
                                collate_fn=collate_simple_mention_ranker)
    ctr = count()
    for batch in progressbar(conll_test_set):
      (num_candidates, features), target_rankings = batch
      target = [ranking[0] for ranking in target_rankings]
      candidate_scores = model(features)
      top_1 = []
      offset = 0
      for ranking_size in num_candidates:
        top_1.append(torch.argmax(candidate_scores[offset : offset + ranking_size]).item())
        offset += ranking_size
      for idx, guess, label in zip(ctr, guess, target):
        if guess == label:
          num_correct += 1
        else:
          missed_idxs.append(idx)
          guessed_when_missed.append(guess)
    with open('./missed_idxs', 'w') as fh:
      fh.writelines([str(dataset[idx]) for idx in missed_idxs])
    with open('./guessed_when_missed', 'w') as fh:
      fh.writelines([str(idx) for idx in guessed_when_missed])



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
