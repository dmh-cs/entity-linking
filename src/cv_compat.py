from collections import Counter
from heapq import nlargest
from itertools import count, groupby
from operator import itemgetter
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
from scipy.sparse import load_npz, csr_matrix

import numpy as np
from nltk.stem.snowball import SnowballStemmer

from simple_conll_dataset import SimpleCoNLLDataset, collate_simple_mention_ranker
from parsers import parse_text_for_tokens
from data_fetchers import get_connection
from fixed_weights_model import FixedWeights
from ltr_bow import get_model
from samplers import SubsetSequentialSampler
from max_product import emissions_from_flat_scores, compatibilities_from_ids, mp_shallow_tree_doc
from create_compats import MentionCoNLLDataset, get_idf, sparse_to_tfidf_vs

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
      doc_id_dataset = MentionCoNLLDataset(cursor,
                                           './AIDA-YAGO2-dataset.tsv',
                                           p.run.lookups_path,
                                           p.train.train_size)
      dataset = SimpleCoNLLDataset(cursor,
                                   token_idx_lookup,
                                   full_token_idx_lookup,
                                   conll_path,
                                   p.run.lookups_path,
                                   p.run.idf_path,
                                   p.train.train_size,
                                   p.run.txt_dataset_path)
      # compats = load_npz('compats_wiki+conll_100000.npz')
      with open('./entity_to_row_id.pkl', 'rb') as fh:
        entity_id_to_row = pickle.load(fh)
      idf = get_idf(token_idx_lookup, p.run.idf_path)
      desc_fs_sparse = csr_matrix(load_npz('./desc_fs.npz'))
      desc_vs = csr_matrix(sparse_to_tfidf_vs(idf, desc_fs_sparse))
      norm = np.sqrt((desc_vs.multiply(desc_vs)).sum(1))
      ctr = count()
      num_correct = 0
      num_in_val = 0
      num_correct_small = 0
      num_in_val_small = 0
      grouped = groupby(((dataset[idx], doc_id_dataset.mention_doc_id[idx])
                         for idx in range(len(val_indices) + len(test_indices))),
                        key=itemgetter(1))
      batches = [collate_simple_mention_ranker([data for data, doc_id in g])
                 for doc_id, g in grouped]
      val_indices = set(val_indices)
      for document_batch in progressbar(batches):
        (candidate_ids, features), target_rankings = document_batch
        target = [ranking[0] for ranking in target_rankings]
        candidate_scores = model(features)
        emissions = emissions_from_flat_scores([len(ids) for ids in candidate_ids],
                                               candidate_scores)
        keep_top_n = 5
        top_emissions = []
        top_cands = []
        idxs_to_check = []
        for i, (emission, cand_ids, top_1, idx) in enumerate(zip(emissions, candidate_ids, target, ctr)):
          if len(cand_ids) > 1:
            if cand_ids[np.argmax(emission)] != top_1:
              if idx in val_indices:
                idxs_to_check.append(i)

          em, cand = zip(*nlargest(keep_top_n, zip(emission, cand_ids), key=itemgetter(0)))
          top_emissions.append(np.array(em))
          top_cands.append(cand)
        compatibilities = compatibilities_from_ids(entity_id_to_row,
                                                   desc_vs,
                                                   norm,
                                                   top_cands)
        top_1_idx = mp_shallow_tree_doc(top_emissions,
                                        compatibilities)
        top_1 = [cand_ids[idx] for cand_ids, idx in zip(top_cands, top_1_idx)]
        for guess, label in zip(top_1, target):
          num_in_val += 1
          if guess == label: num_correct += 1
        for idx in idxs_to_check:
          guess = top_1[idx]
          label = target[idx]
          num_in_val_small += 1
          if guess == label: num_correct_small += 1
      print(num_correct / num_in_val)
      print(num_correct_small / num_in_val_small)



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
