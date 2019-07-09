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
from scipy.sparse import load_npz

import numpy as np
from nltk.stem.snowball import SnowballStemmer

from simple_conll_dataset import SimpleCoNLLDataset, collate_simple_mention_ranker
from parsers import parse_text_for_tokens
from data_fetchers import get_connection
from fixed_weights_model import FixedWeights
from ltr_bow import get_model
from samplers import SubsetSequentialSampler
from max_product import emissions_from_flat_scores, compatibility_from_ids, mp_doc

from rabbit_ml import get_cli_args

from args_config import args


def load_model(model_params, train_params):
  if model_params.just_tfidf:
    return FixedWeights([1] + [0] * 12)
  elif model_params.just_str_sim:
    return FixedWeights([0, 0, 1] + [0] * 10)
  elif model_params.just_wiki2vec:
    return FixedWeights([0] * 8 + [0, 0, 1, 0, 0])
  else:
    model = get_model(model_params, train_params)
    train_str = 'pairwise' if train_params.use_pairwise else ''
    train_str += '_{}_'.format(train_params.dropout_keep_prob)
    train_str += '_{}_'.format(train_params.learning_rate)
    loss_str = 'hinge_{}'.format(train_params.margin) if train_params.use_hinge else ''
    loss_str += '_{}_'.format(train_params.margin) if train_params.use_hinge else ''
    path = './ltr_model_' + ','.join(str(sz) for sz in model_params.hidden_sizes) + train_str + '_' + loss_str
    model.load_state_dict(torch.load(path))
    return model

def main():
  p = get_cli_args(args)
  conll_path = 'custom.tsv' if p.run.use_custom else './AIDA-YAGO2-dataset.tsv'
  db_connection = get_connection(p.run.env_path)
  model = load_model(p.model, p.train)
  with open('./tokens.pkl', 'rb') as fh: token_idx_lookup = pickle.load(fh)
  with open('./glove_token_idx_lookup.pkl', 'rb') as fh: full_token_idx_lookup = pickle.load(fh)
  with open('./val_test_indices.json', 'r') as fh:
    val_indices, test_indices = json.load(fh)
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
                                  batch_sampler=BatchSampler(SubsetSequentialSampler(val_indices),
                                                             p.run.batch_size,
                                                             False),
                                  collate_fn=collate_simple_mention_ranker)
      compats = load_npz('compats_wiki+conll_100000.npz')
      with open('./entity_to_row_id.pkl', 'rb') as fh:
        entity_id_to_row = pickle.load(fh)
      results = {}
      for emission_weight in np.arange(0.0, 1.0, 0.1):
        ctr = count()
        num_correct = 0
        for document_batch in progressbar(conll_test_set):
          (candidate_ids, features), target_rankings = document_batch
          target = [ranking[0] for ranking in target_rankings]
          candidate_scores = model(features)
          emissions = emissions_from_flat_scores([len(ids) for ids in candidate_ids],
                                                 candidate_scores)
          compatibilities = compatibility_from_ids(entity_id_to_row,
                                                   compats,
                                                   candidate_ids)
          top_1 = mp_doc(emissions,
                         compatibilities,
                         emission_weight=emission_weight)
          for guess, label, idx in zip(top_1, target, ctr):
            if guess == label: num_correct += 1
        results[emission_weight] = num_correct / next(ctr)
        print(emission_weight)
        print(num_correct / next(ctr))
      print(results)



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
