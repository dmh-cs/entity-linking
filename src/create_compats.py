from args_config import args
from cache import read_cache
from coll_transformers import DefaultVal
from collections import Counter
from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior_cnts, get_candidate_strs, get_str_lookup, get_embedding_dict
from data_transformers import get_mention_sentences_from_infos, pad_batch_list
from db_backed_bow import DBBoW
from doc_lookup import DocLookup
from dotenv import load_dotenv
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.snowball import SnowballStemmer
from parsers import parse_text_for_tokens
from progressbar import progressbar
from rabbit_ml import get_cli_args
import Levenshtein
import ast
import json
import nltk
import numpy as np
import os
import pickle
import pydash as _
import pymysql.cursors
import re
import torch
import utils as u
from scipy.sparse import save_npz, dok_matrix, csr_matrix, coo_matrix, load_npz
from itertools import combinations

class MentionDataset:
  def __init__(self,
               cursor,
               lookups_path,
               train_size):
    self.cursor = cursor
    lookups = load_entity_candidate_ids_and_label_lookup(lookups_path, train_size)
    label_to_entity_id = _.invert(lookups['entity_labels'])
    self.entity_candidates_prior = {entity_text: {label_to_entity_id[label]: candidates
                                                  for label, candidates in prior.items()}
                                    for entity_text, prior in lookups['entity_candidates_prior'].items()}
    self.prior_approx_mapping = u.get_prior_approx_mapping(self.entity_candidates_prior)
    self.mentions = None
    self.labels = None

  def __len__(self):
    return len(self.with_labels)

  def __getitem__(self, idx):
    label = self.labels[idx]
    mention = self.mentions[idx]
    cands = get_candidate_ids_simple(self.entity_candidates_prior,
                                     self.prior_approx_mapping,
                                     mention).tolist()
    if label not in cands: return
    return cands

class MentionWikiDataset(MentionDataset):
  def __init__(self,
               cursor,
               page_ids,
               lookups_path,
               train_size):
    super().__init__(cursor,
                     lookups_path,
                     train_size)
    self.cursor = cursor
    self.page_ids = page_ids
    self.mention_infos = self.get_mention_infos(page_ids)
    self.mentions = [info['mention'] for info in self.mention_infos]
    self.labels = [info['entity_id'] for info in self.mention_infos]

  def get_mention_infos(self, page_ids):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id in (' + str(page_ids)[1:-1] + ')')
    return self.cursor.fetchall()

class MentionCoNLLDataset(MentionDataset):
  def __init__(self,
               cursor,
               conll_path,
               lookups_path,
               train_size):
    super().__init__(cursor,
                     lookups_path,
                     train_size)
    with open(conll_path, 'r') as fh:
      lines = fh.read().strip().split('\n')[:-1]
    self.mentions = get_mentions(lines)
    self.entity_page_ids = get_entity_page_ids(lines)
    self.labels = from_page_ids_to_entity_ids(self.cursor, self.entity_page_ids)

def sparse_to_tfidf_vs(idf, sparse):
  idf_array = np.array([idf[i] if i in idf else idf[0]
                        for i in range(sparse.shape[1])])
  return sparse.multiply(idf_array.reshape(1, -1))

def get_idf(token_idx_lookup, idf_path):
  with open(idf_path) as fh:
    idf = json.load(fh)
  lookup = {u.to_idx(token_idx_lookup, token): token_idf
            for token, token_idf in idf.items()}
  lookup[token_idx_lookup['<UNK>']] = 0
  lookup[token_idx_lookup['<PAD>']] = 0
  return lookup

def main():
  p = get_cli_args(args)
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

    datasets = [MentionCoNLLDataset(cursor,
                                    './AIDA-YAGO2-dataset.tsv',
                                    p.run.lookups_path,
                                    p.train.train_size),
                MentionWikiDataset(cursor,
                                   page_ids,
                                   p.run.lookups_path,
                                   p.train.train_size)]
    with open('./entity_to_row_id.pkl', 'rb') as fh:
      entity_id_to_row = pickle.load(fh)
    idf = get_idf(token_idx_lookup, p.run.idf_path)
    desc_fs_sparse = csr_matrix(load_npz('./desc_fs.npz'))
    # mat = dok_matrix((len(entity_id_to_row), len(entity_id_to_row)))
    desc_vs = csr_matrix(sparse_to_tfidf_vs(idf, desc_fs_sparse))
    norm = (desc_vs.multiply(desc_vs)).sum(1)
    all_e_ids = set()
    data = []
    i = []
    j = []
    for dataset in datasets:
      for cands in progressbar(iter(dataset)):
        if cands is None: continue
        cand_rows = [entity_id_to_row[e_id]
                     for e_id in cands
                     if (e_id in entity_id_to_row) and (e_id not in all_e_ids)]
        all_e_ids.update(cands)
        cand_mat = desc_vs[cand_rows]
        scores = cand_mat.dot(cand_mat.T) / norm[cand_rows]
        data.extend(np.array(scores).ravel().tolist())
        i.extend(cand_rows * len(cand_rows))
        j.extend([row_num
                  for row_num in cand_rows
                  for __ in range(len(cand_rows))])
        # for i, row_1 in enumerate(cand_rows):
        #   for j, row_2 in enumerate(cand_rows):
        #     mat[row_1, row_2] = scores[i, j]
    mat = csr_matrix(coo_matrix((data, (i, j))))
    train_str = 'wiki+conll_' + '_'.join([str(p.train.num_pages_to_use)])
    save_npz('compats_{}.npz'.format(train_str), mat)



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
