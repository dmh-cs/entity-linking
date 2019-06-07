import pydash as _
import Levenshtein
from collections import Counter
import torch
from torch.utils.data import Dataset
from nltk.stem.snowball import SnowballStemmer
import json
import re

from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior_cnts, get_candidate_strs
from parsers import parse_text_for_tokens
import utils as u
from data_transformers import get_mention_sentences_from_infos, pad_batch_list
from simple_dataset import SimpleDataset

class SimpleCoNLLDataset(SimpleDataset):
  def __init__(self, cursor, conll_path, lookups_path, idf_path, train_size):
    super().__init__(cursor, lookups_path, idf_path, train_size)
    with open(conll_path, 'r') as fh:
      lines = fh.read().strip().split('\n')[:-1]
    self.documents = get_documents(lines)
    self.mentions = get_mentions(lines)
    self.entity_page_ids = get_entity_page_ids(lines)
    self.labels = from_page_ids_to_entity_ids(self.cursor, self.entity_page_ids)
    self.mention_doc_id = get_doc_id_per_mention(lines)
    self.mentions_by_doc_id = get_mentions_by_doc_id(lines)
    self.mention_sentences = get_mention_sentences(self.documents, self.mentions)
    self.document_lookup = self.documents
    self.mention_fs = [self._to_f(sentence) for sentence in self.mention_sentences]
    self.page_f_lookup = [self._to_f(parse_text_for_tokens(doc)) for doc in self.document_lookup]
    self._post_init()

def collate_simple_mention_ranker(batch):
  element_features = []
  target_rankings = []
  features, candidate_ids, labels = zip(*batch)
  for mention_features, mention_candidate_ids, label in zip(features, candidate_ids, labels):
    ranking = [label]
    features_for_ranking = []
    for candidate_features, candidate_id in zip(mention_features, mention_candidate_ids):
      if candidate_id != label:
        ranking.append(candidate_id)
      features_for_ranking.append(candidate_features)
    target_rankings.append(ranking)
    element_features.append(features_for_ranking)
  flattened_features = _.flatten(element_features)
  return (candidate_ids, torch.tensor(flattened_features)), target_rankings
