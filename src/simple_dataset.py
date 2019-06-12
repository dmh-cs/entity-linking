import pydash as _
import Levenshtein
from collections import Counter
import torch
from torch.utils.data import Dataset
from nltk.stem.snowball import SnowballStemmer
import json
import re

from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior_cnts, get_candidate_strs, get_str_lookup
from parsers import parse_text_for_tokens
import utils as u
from data_transformers import get_mention_sentences_from_infos, pad_batch_list
from cache import read_cache
from db_backed_bow import DBBoW
from coll_transformers import DefaultVal

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
  def __init__(self, cursor, lookups_path, idf_path, train_size):
    with open(idf_path) as fh:
      self.idf = json.load(fh)
    self.cursor = cursor
    self.query_template = 'select e.id as entity_id, left(p.content, 2000) as text from entities e join pages p on e.text = p.title where e.id = {}'
    self.desc_fs = DefaultVal(DBBoW('desc', self.cursor, self.query_template),
                              {})
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
    self.page_f_lookup = None
    self.with_labels = None
    self._candidate_strs_lookup = read_cache('./candidate_strs_lookup.pkl',
                                             lambda: get_str_lookup(cursor))

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

  def _to_f(self, tokens):
    return dict(Counter(self.stemmer.stem(token)
                        for token in tokens))

  def calc_tfidf(self, candidate_f, mention_f):
    return sum(cnt * candidate_f.get(token, 0) * self.idf.get(token,
                                                         self.idf.get(token.lower(), 0.0))
               for token, cnt in mention_f.items())

  def __len__(self): return len(self.with_labels)

  def __getitem__(self, idx):
    i = self.with_labels[idx]
    label = self.labels[i]
    mention = self.mentions[i]
    mention_f = self.mention_fs[i]
    mention_doc_id = self.mention_doc_id[i]
    page_f = self.page_f_lookup[mention_doc_id]
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
    cands_with_page = []
    for candidate_raw_features in zip(candidate_ids,
                                      candidate_mention_sim,
                                      prior):
      candidate_id, candidate_mention_sim, candidate_prior = candidate_raw_features
      if candidate_id not in candidate_fs: continue
      cands_with_page.append(candidate_id)
      candidate_f = candidate_fs[candidate_id]
      mention_tfidf = self.calc_tfidf(candidate_f, mention_f)
      page_tfidf = self.calc_tfidf(candidate_f, page_f)
      all_mentions_features.append([mention_tfidf,
                                    sum(candidate_f.values()),
                                    sum(mention_f.values()),
                                    page_tfidf,
                                    sum(page_f.values()),
                                    candidate_mention_sim,
                                    candidate_prior,
                                    times_mentioned])
    return all_mentions_features, cands_with_page, label
