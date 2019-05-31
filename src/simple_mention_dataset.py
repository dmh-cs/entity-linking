import pydash as _
import Levenshtein
from collections import Counter
import torch
import pickle
from nltk.stem.snowball import SnowballStemmer
import json

from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior_cnts, get_candidate_strs
from parsers import parse_text_for_tokens
from data_transformers import get_mention_sentences_from_infos
import utils as u

class SimpleMentionDataset():
  def __init__(self, cursor, page_ids, lookups_path, idf_path, train_size):
    self.stemmer = SnowballStemmer('english')
    with open(idf_path) as fh:
      self.idf = json.load(fh)
    self.page_content_lim = 2000
    self.cursor = cursor
    self.page_ids = page_ids
    self.document_lookup = self.get_document_lookup(page_ids)
    self.mention_infos = self.get_mention_infos(page_ids)
    self.mentions = [info['mention'] for info in self.mention_infos]
    self.labels = [info['entity_id'] for info in self.mention_infos]
    self.mention_doc_id = [info['page_id'] for info in self.mention_infos]
    self.with_label = [i for i, x in enumerate(self.labels) if x != -1]
    self.mention_sentences = get_mention_sentences_from_infos(self.document_lookup, self.mention_infos)
    self.mention_fs = [dict(Counter(self.stemmer.stem(token)
                                    for token in parse_text_for_tokens(sentence)))
                       for sentence in self.mention_sentences]
    self.page_f_lookup = {page_id: dict(Counter(self.stemmer.stem(token)
                                                         for token in parse_text_for_tokens(doc[:self.page_content_lim])))
                                   for page_id, doc in self.document_lookup.items()}
    lookups = load_entity_candidate_ids_and_label_lookup(lookups_path, train_size)
    label_to_entity_id = _.invert(lookups['entity_labels'])
    self.entity_candidates_prior = {entity_text: {label_to_entity_id[label]: candidates
                                                  for label, candidates in prior.items()}
                                    for entity_text, prior in lookups['entity_candidates_prior'].items()}
    self.prior_approx_mapping = u.get_prior_approx_mapping(self.entity_candidates_prior)

  def get_document_lookup(self, page_ids):
    self.cursor.execute(f'select id, left(content, {self.page_content_lim}) as content from pages where id in (' + str(page_ids)[1:-1] + ')')
    return {row['id']: row['content'] for row in self.cursor.fetchall()}

  def get_mention_infos(self, page_ids):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id in (' + str(page_ids)[1:-1] + ')')
    return self.cursor.fetchall()

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
    prior, total = get_p_prior_cnts(self.entity_candidates_prior,
                                    self.prior_approx_mapping,
                                    mention,
                                    candidate_ids)
    candidate_mention_sim = [Levenshtein.ratio(mention, candidate_str)
                             for candidate_str in candidate_strs]
    all_mentions_features = []
    for candidate_raw_features in zip(candidate_ids,
                                      candidate_mention_sim,
                                      prior,
                                      total):
      candidate_id, candidate_mention_sim, candidate_prior, candidate_total = candidate_raw_features
      candidate_f = self.page_f_lookup[candidate_id]
      mention_tfidf = self.calc_tfidf(candidate_f, mention_f)
      page_tfidf = self.calc_tfidf(candidate_f, page_f)
      all_mentions_features.append([mention_tfidf,
                                    page_tfidf,
                                    candidate_mention_sim,
                                    candidate_prior,
                                    candidate_total])
    return all_mentions_features, label

def collate_simple_mention(batch):
  features, labels = zip(*batch)
  return torch.tensor(pad_batch_list(0, features)), torch.tensor(labels)
