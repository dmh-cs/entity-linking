import pydash as _
import Levenshtein
from collections import Counter
import torch
from torch.utils.data import Dataset
from nltk.stem.snowball import SnowballStemmer
import json
from pymongo import MongoClient
import re

from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior_cnts, get_candidate_strs
from parsers import parse_text_for_tokens
import utils as u
from data_transformers import get_mention_sentences_from_infos, pad_batch_list


def get_desc_fs(pages_db, cursor, stemmer, cand_ids):
  cursor.execute('select entity_id, source_id from entity_by_page where entity_id in (' + str(cand_ids)[1:-1] + ')')
  entity_source_id_lookup = {row['entity_id']: row['source_id'] for row in cursor.fetchall()}
  fs = {}
  for entity_id, source_id in entity_source_id_lookup.items():
    page_content = pages_db.find_one({'pageID': str(source_id)})['plaintext'][:2000]
    tokenized = parse_text_for_tokens(page_content)
    fs[entity_id] = dict(Counter(stemmer.stem(token) for token in tokenized))
  return fs

def clean_entity_text(entity_text):
  return re.sub(r'\s*\(.*\)$', '', entity_text)

class SimpleCoNLLDataset(Dataset):
  def __init__(self, cursor, conll_path, lookups_path, idf_path, train_size):
    with open(idf_path) as fh:
      self.idf = json.load(fh)
    with open(conll_path, 'r') as fh:
      lines = fh.read().strip().split('\n')[:-1]
    client = MongoClient()
    dbname = 'enwiki'
    db = client[dbname]
    self.pages_db = db['pages']
    self.cursor = cursor
    self.documents = get_documents(lines)
    self.mentions = get_mentions(lines)
    self.entity_page_ids = get_entity_page_ids(lines)
    self.labels = from_page_ids_to_entity_ids(self.cursor, self.entity_page_ids)
    self.mention_doc_id = get_doc_id_per_mention(lines)
    self.mentions_by_doc_id = get_mentions_by_doc_id(lines)
    self.mention_sentences = get_mention_sentences(self.documents, self.mentions)
    self.stemmer = SnowballStemmer('english')
    self.document_lookup = self.documents
    self.with_label = [i for i, x in enumerate(self.labels) if x != -1]
    self.mention_fs = [dict(Counter(self.stemmer.stem(token)
                                    for token in sentence))
                       for sentence in self.mention_sentences]
    self.page_f_lookup = {page_id: dict(Counter(self.stemmer.stem(token)
                                                for token in parse_text_for_tokens(doc)))
                                   for page_id, doc in self.document_lookup}
    lookups = load_entity_candidate_ids_and_label_lookup(lookups_path, train_size)
    label_to_entity_id = _.invert(lookups['entity_labels'])
    self.entity_candidates_prior = {entity_text: {label_to_entity_id[label]: candidates
                                                  for label, candidates in prior.items()}
                                    for entity_text, prior in lookups['entity_candidates_prior'].items()}
    self.prior_approx_mapping = u.get_prior_approx_mapping(self.entity_candidates_prior)

  def calc_tfidf(self, candidate_f, mention_f):
    return sum(cnt * candidate_f.get(token, 0) * self.idf.get(token,
                                                         self.idf.get(token.lower(), 0.0))
               for token, cnt in mention_f.items())

  def __getitem__(self, idx):
    label = self.labels[idx]
    mention = self.mentions[idx]
    mention_f = self.mention_fs[idx]
    mention_doc_id = self.mention_doc_id[idx]
    page_f = self.page_f_lookup[mention_doc_id]
    candidate_ids = get_candidate_ids_simple(self.entity_candidates_prior,
                                             self.prior_approx_mapping,
                                             mention).tolist()
    candidate_strs = get_candidate_strs(self.cursor, candidate_ids)
    prior = get_p_prior_cnts(self.entity_candidates_prior,
                             self.prior_approx_mapping,
                             mention,
                             candidate_ids)
    times_mentioned = sum(prior)
    candidate_mention_sim = [Levenshtein.ratio(mention, clean_entity_text(candidate_str))
                             for candidate_str in candidate_strs]
    all_mentions_features = []
    candidate_fs = get_desc_fs(self.pages_db, self.cursor, self.stemmer, candidate_ids)
    for candidate_raw_features in zip(candidate_ids,
                                      candidate_mention_sim,
                                      prior):
      candidate_id, candidate_mention_sim, candidate_prior = candidate_raw_features
      candidate_f = candidate_fs[candidate_id]
      mention_tfidf = self.calc_tfidf(candidate_f, mention_f)
      page_tfidf = self.calc_tfidf(candidate_f, page_f)
      all_mentions_features.append([mention_tfidf,
                                    page_tfidf,
                                    candidate_mention_sim,
                                    candidate_prior,
                                    times_mentioned])
    return all_mentions_features, candidate_ids, label

def collate_simple_mention_ranker(batch):
  element_features = []
  target_rankings = []
  features, candidate_ids, labels = zip(*batch)
  for mention_features, mention_candidate_ids, label in zip(features, candidate_ids, labels):
    target_idx = mention_candidate_ids.index(label)
    target_features = features[target_idx]
    ranking = [label]
    features_for_ranking = [target_features]
    for candidate_features, candidate_id in zip(mention_features, mention_candidate_ids):
      if candidate_id != label:
        ranking.append(candidate_id)
        features_for_ranking.append(candidate_features)
    target_rankings.append(ranking)
    element_features.append(features_for_ranking)
  num_candidates = [len(to_rank) for to_rank in element_features]
  flattened_features = _.flatten(element_features)
  return (num_candidates, torch.tensor(flattened_features)), torch.tensor(target_rankings)
