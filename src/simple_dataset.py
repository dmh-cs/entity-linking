from pyrsistent import pvector
import pickle
import pydash as _
import Levenshtein
from collections import Counter
import torch
from torch.utils.data import Dataset
from nltk.stem.snowball import SnowballStemmer
import json
import re
import ast

import nltk
from nltk.corpus import stopwords as nltk_stopwords
import numpy as np

from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior_cnts, get_candidate_strs, get_str_lookup, get_embedding_dict
from parsers import parse_text_for_tokens
import utils as u
from data_transformers import get_mention_sentences_from_infos, pad_batch_list
from cache import read_cache
from db_backed_bow import DBBoW
from coll_transformers import DefaultVal
from doc_lookup import DocLookup

def clean_entity_text(entity_text):
  return re.sub(r'\s*\(.*\)$', '', entity_text)

def _get_desc_fs(cursor):
  stemmer = SnowballStemmer('english')
  cursor.execute('select e.id as entity_id, left(p.content, 2000) as text from entities e join pages p on e.text = p.title')
  entity_desc_bow = {}
  for row in cursor.fetchall():
    tokens = parse_text_for_tokens(row['text'])
    entity_desc_bow[row['entity_id']] = dict(Counter(stemmer.stem(token) for token in tokens))
  return entity_desc_bow

class SimpleDataset(Dataset):
  def __init__(self,
               cursor,
               token_idx_lookup,
               full_token_idx_lookup,
               lookups_path,
               idf_path,
               train_size,
               txt_dataset_path,
               pkl_dataset_prefix=None):
    self.txt_dataset_path = txt_dataset_path
    self.pkl_dataset_prefix = pkl_dataset_prefix
    if self.pkl_dataset_prefix is not None:
      self.current_part = None
      return
    if self.txt_dataset_path is not None:
      if '.pkl' in self.txt_dataset_path:
        with open(self.txt_dataset_path, 'rb') as fh:
          self.dataset_cache = pickle.load(fh)
          return
      with open(self.txt_dataset_path) as fh:
        self.dataset_cache = [ast.literal_eval(line) for line in fh.readlines()]
        return
    with open(idf_path) as fh:
      self.idf = json.load(fh)
    self.cursor = cursor
    with open('./entity_to_row_id.pkl', 'rb') as fh: entity_id_to_row = pickle.load(fh)
    self.desc_fs = DocLookup('./desc_fs.npz',
                             entity_id_to_row,
                             token_idx_mapping=_.invert(token_idx_lookup),
                             default_value={},
                             use_default=True)
    self.desc_fs_unstemmed = DocLookup('./desc_unstemmed_fs.npz',
                                       entity_id_to_row,
                                       token_idx_mapping=_.invert(full_token_idx_lookup),
                                       default_value={'<PAD>': 1},
                                       use_default=True)
    self.embedding_dict = get_embedding_dict('./glove.6B.300d.txt',
                                             embedding_dim=300)
    self.stemmer = SnowballStemmer('english')
    lookups = load_entity_candidate_ids_and_label_lookup(lookups_path, train_size)
    label_to_entity_id = _.invert(lookups['entity_labels'])
    self.entity_candidates_prior = {entity_text: {label_to_entity_id[label]: candidates
                                                  for label, candidates in prior.items()}
                                    for entity_text, prior in lookups['entity_candidates_prior'].items()}
    self.prior_approx_mapping = u.get_prior_approx_mapping(self.entity_candidates_prior)
    self.mentions = None
    self.labels = None
    self.mention_doc_id = None
    self.mention_sentences = None
    self.mention_fs = None
    self.mention_fs_unstemmed = None
    self.page_f_lookup = None
    self.with_labels = None
    self._candidate_strs_lookup = read_cache('./candidate_strs_lookup.pkl',
                                             lambda: get_str_lookup(cursor))
    self.stopwords = set(nltk_stopwords.words('english'))

  def _post_init(self):
    self.with_labels = []
    num_without_cand = 0
    for idx, (mention, label) in enumerate(zip(self.mentions, self.labels)):
      if label in get_candidate_ids_simple(self.entity_candidates_prior,
                                           self.prior_approx_mapping,
                                           mention).tolist():
        self.with_labels.append(idx)
      else:
        num_without_cand += 1
    print('num without candidates:', num_without_cand)
    print('num with candidates:', len(self.with_labels))

  def _to_f(self, tokens, stem_p=True):
    if stem_p:
      return dict(Counter(self.stemmer.stem(token)
                          for token in tokens))
    else:
      return dict(Counter(tokens))

  def calc_tfidf(self, candidate_f, mention_f):
    return sum(cnt * candidate_f.get(token, 0) * self.idf.get(token,
                                                              self.idf.get(token.lower(), 0.0)) ** 2
               for token, cnt in mention_f.items())

  def calc_tfidf_norm(self, f):
    return sum((cnt * self.idf.get(token,
                                   self.idf.get(token.lower(), 0.0))) ** 2
               for token, cnt in f.items())

  def __len__(self):
    if self.txt_dataset_path is not None:
      return len(self.dataset_cache)
    else:
      return len(self.with_labels)

  def _f_to_vec(self, f_unstemmed):
    cnts = f_unstemmed.values()
    idfs = [self.idf.get(token, self.idf.get(token.lower()))
            for token in f_unstemmed.keys()]
    vecs = [self.embedding_dict.get(token, self.embedding_dict.get(token.lower()))
            for token in f_unstemmed.keys()]
    return torch.sum(torch.stack([cnt * idf * vec
                                  if (token not in self.stopwords) and all(x is not None
                                                                           for x in (token, cnt, idf, vec))
                                  else self.embedding_dict['<PAD>']
                                  for token, cnt, idf, vec in zip(f_unstemmed.keys(), cnts, idfs, vecs)]),
                     dim=0)

  def _idx_in_part(self, idx):
    return np.searchsorted(range(100000, 2140542 // 100000), idx // 100000)

  def __getitem__(self, idx):
    if self.pkl_dataset_prefix is not None:
      if self._idx_in_part(idx) != self.current_part:
        self.current_part = self._idx_in_part(idx)
        with open(self.pkl_dataset_prefix + '_{}'.format(self.current_part) + '.pkl', 'rb') as fh:
          self.targets = []
          self.cands = []
          for t, c in pickle.load(fh):
            self.targets.extend(t)
            self.cands.extend(c)
      return (self.targets[idx % 100000], self.cands[idx % 100000]), 0.0
    if self.txt_dataset_path is not None: return self.dataset_cache[idx]
    i = self.with_labels[idx]
    label = self.labels[i]
    mention = self.mentions[i]
    mention_f = self.mention_fs[i]
    mention_f_unstemmed = self.mention_fs_unstemmed[i]
    mention_doc_id = self.mention_doc_id[i]
    page_f = self.page_f_lookup[mention_doc_id]
    page_f_unstemmed = self.page_f_lookup_unstemmed[mention_doc_id]
    mention_vec = self._f_to_vec(mention_f_unstemmed)
    page_vec = self._f_to_vec(page_f_unstemmed)
    candidate_ids = get_candidate_ids_simple(self.entity_candidates_prior,
                                             self.prior_approx_mapping,
                                             mention).tolist()
    candidate_strs = [self._candidate_strs_lookup[cand_id] for cand_id in candidate_ids]
    prior = get_p_prior_cnts(self.entity_candidates_prior,
                             self.prior_approx_mapping,
                             mention,
                             candidate_ids)
    times_mentioned = sum(prior)
    candidate_mention_sim = [Levenshtein.ratio(mention, clean_entity_text(candidate_str))
                             for candidate_str in candidate_strs]
    all_mentions_features = []
    candidate_fs = {cand_id: fs
                    for cand_id, fs in zip(candidate_ids, self.desc_fs[candidate_ids])}
    candidate_fs_unstemmed = {cand_id: fs
                              for cand_id, fs in zip(candidate_ids,
                                                     self.desc_fs_unstemmed[candidate_ids])}
    cands_with_page = []
    for candidate_raw_features in zip(candidate_ids,
                                      candidate_mention_sim,
                                      prior):
      candidate_id, candidate_mention_sim, candidate_prior = candidate_raw_features
      if candidate_id not in candidate_fs: continue
      cands_with_page.append(candidate_id)
      candidate_f = candidate_fs[candidate_id]
      candidate_f_unstemmed = candidate_fs_unstemmed[candidate_id]
      cand_vec = self._f_to_vec(candidate_f_unstemmed)
      mention_wiki2vec_dot = cand_vec.dot(mention_vec).item()
      mention_wiki2vec_dot_unit = (mention_wiki2vec_dot / (cand_vec.norm() * mention_vec.norm())).item()
      mention_wiki2vec_dot_unit = 0.0 if torch.isnan(torch.tensor(mention_wiki2vec_dot_unit)) else mention_wiki2vec_dot_unit
      page_wiki2vec_dot = cand_vec.dot(page_vec).item()
      page_wiki2vec_dot_unit = (page_wiki2vec_dot / (cand_vec.norm() * page_vec.norm())).item()
      page_wiki2vec_dot_unit = 0.0 if torch.isnan(torch.tensor(page_wiki2vec_dot_unit)) else page_wiki2vec_dot_unit
      mention_tfidf = self.calc_tfidf(candidate_f, mention_f)
      candidate_tfidf_norm = self.calc_tfidf_norm(candidate_f)
      page_tfidf = self.calc_tfidf(candidate_f, page_f)
      all_mentions_features.append([mention_tfidf,
                                    sum(candidate_f.values()),
                                    sum(mention_f.values()),
                                    page_tfidf,
                                    sum(page_f.values()),
                                    candidate_mention_sim,
                                    candidate_prior,
                                    times_mentioned,
                                    mention_wiki2vec_dot,
                                    page_wiki2vec_dot,
                                    mention_wiki2vec_dot_unit,
                                    page_wiki2vec_dot_unit,
                                    candidate_tfidf_norm])
    return all_mentions_features, cands_with_page, label
