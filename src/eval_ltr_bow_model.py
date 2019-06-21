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
import pickle

import numpy as np
from nltk.stem.snowball import SnowballStemmer

from simple_conll_dataset import SimpleCoNLLDataset, collate_simple_mention_ranker
from parsers import parse_text_for_tokens
from data_fetchers import get_connection
from fixed_weights_model import FixedWeights
from ltr_bow import LtRBoW

from rabbit_ml import get_cli_args

from args_config import args


def load_model(model_params, train_params):
  if model_params.just_tfidf:
    return FixedWeights([1] + [0] * 11)
  elif model_params.just_str_sim:
    return FixedWeights([0, 0, 1] + [0] * 9)
  elif model_params.just_wiki2vec:
    return FixedWeights([0] * 8 + [0, 1, 0, 0])
  else:
    model = LtRBoW(model_params.hidden_sizes, dropout_keep_prob=train_params.dropout_keep_prob)
    train_str = 'pairwise' if train_params.use_pairwise else ''
    loss_str = 'hinge_{}'.format(train_params.margin) if train_params.use_hinge else ''
    path = './ltr_model_' + ','.join(str(sz) for sz in model_params.hidden_sizes) + train_str + '_' + loss_str
    model.load_state_dict(torch.load(path))
    return model

def main():
  p = get_cli_args(args)
  conll_path = 'custom.tsv' if p.run.use_custom else './AIDA-YAGO2-dataset.tsv'
  num_correct = 0
  missed_idxs = []
  guessed_when_missed = []
  db_connection = get_connection(p.run.env_path)
  model = load_model(p.model, p.train)
  with open('./tokens.pkl', 'rb') as fh: token_idx_lookup = pickle.load(fh)
  with open('./glove_token_idx_lookup.pkl', 'rb') as fh: full_token_idx_lookup = pickle.load(fh)
  model.eval()
  with torch.no_grad():
    with db_connection.cursor() as cursor:
      dataset = SimpleCoNLLDataset(cursor,
                                   token_idx_lookup,
                                   full_token_idx_lookup,
                                   conll_path,
                                   p.run.lookups_path,
                                   p.run.idf_path,
                                   p.train.train_size,
                                   p.run.txt_dataset_path)
      conll_test_set = DataLoader(dataset,
                                  batch_sampler=BatchSampler(SequentialSampler(dataset),
                                                             p.run.batch_size,
                                                             False),
                                  collate_fn=collate_simple_mention_ranker)
      ctr = count()
      for batch in progressbar(conll_test_set):
        (candidate_ids, features), target_rankings = batch
        target = [ranking[0] for ranking in target_rankings]
        candidate_scores = model(features)
        top_1 = []
        offset = 0
        for ids in candidate_ids:
          ranking_size = len(ids)
          top_1.append(ids[torch.argmax(candidate_scores[offset : offset + ranking_size]).item()])
          offset += ranking_size
        for guess, label, idx, ids in zip(top_1, target, ctr, candidate_ids):
          if guess == label:
            num_correct += 1
          else:
            missed_idxs.append(idx)
            guessed_when_missed.append(guess)
      print(num_correct / next(ctr))
      import ipdb; ipdb.set_trace()
      with open('./missed_idxs', 'w') as fh:
        fh.write('\n'.join([str((idx, dataset[idx])) for idx in missed_idxs]))
      with open('./guessed_when_missed', 'w') as fh:
        fh.write('\n'.join([str(idx) for idx in guessed_when_missed]))



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
