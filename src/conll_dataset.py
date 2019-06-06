from nltk.stem.snowball import SnowballStemmer
import Levenshtein
import re
from collections import defaultdict
import unidecode
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

import pydash as _

from data_transformers import embed_page_content, get_bag_of_nouns
from data_fetchers import get_candidate_ids, get_p_prior, get_candidate_strs
from parsers import parse_for_sentence_spans, parse_for_tokens, parse_text_for_tokens
from conll_helpers import get_documents, get_mentions, get_splits, get_entity_page_ids, from_page_ids_to_entity_ids, get_doc_id_per_mention, get_mentions_by_doc_id, get_mention_sentences
import utils as u

class CoNLLDataset(Dataset):
  def __init__(self,
               cursor,
               entity_candidates_prior,
               embedding,
               token_idx_lookup,
               num_entities,
               num_candidates,
               entity_label_lookup,
               path='./AIDA-YAGO2-dataset.tsv',
               use_wiki2vec=False,
               use_sum_encoder=False):
    self.cursor = cursor
    self.entity_candidates_prior = entity_candidates_prior
    self.embedding = embedding
    self.token_idx_lookup = token_idx_lookup
    self.num_entities = num_entities
    self.num_candidates = num_candidates
    with open(path, 'r') as fh:
      self.lines = fh.read().strip().split('\n')[:-1]
    self.documents = get_documents(self.lines)
    self.embedded_documents = [embed_page_content(self.embedding, self.token_idx_lookup, document)
                               for document in self.documents]
    self.mentions = get_mentions(self.lines)
    self.sentence_splits = get_splits(self.documents, self.mentions)
    self.mention_sentences = get_mention_sentences(self.documents, self.mentions)
    self.entity_page_ids = get_entity_page_ids(self.lines)
    self.labels = from_page_ids_to_entity_ids(cursor, self.entity_page_ids)
    self.with_label = [i for i, x in enumerate(self.labels) if x != -1]
    self.mention_doc_id = get_doc_id_per_mention(self.lines)
    self.mentions_by_doc_id = get_mentions_by_doc_id(self.lines)
    self.entity_label_lookup = entity_label_lookup
    self.entity_id_lookup = {int(label): entity_id for entity_id, label in self.entity_label_lookup.items()}
    self.use_wiki2vec = use_wiki2vec
    self.prior_approx_mapping = u.get_prior_approx_mapping(self.entity_candidates_prior)
    self.use_sum_encoder = use_sum_encoder
    self.stemmer = SnowballStemmer('english')
    self.page_token_cnts_lookup = [dict(Counter(u.to_idx(self.token_idx_lookup, self._stem(token))
                                                for token in parse_text_for_tokens(page_content)))
                                   for page_content in self.documents]

  def _stem(self, text):
    return self.stemmer.stem(text)

  def __len__(self):
    return len(self.with_label)

  def __getitem__(self, idx):
    if self.use_wiki2vec:
      return self._getitem_wiki2vec(idx)
    elif self.use_sum_encoder:
      return self._getitem_sum_encoder(idx)
    else:
      return self._getitem_deep_el(idx)

  def _getitem_deep_el(self, idx):
    idx = self.with_label[idx]
    label = self.entity_label_lookup.get(self.labels[idx], -1)
    mention = self.mentions[idx]
    candidate_ids = get_candidate_ids(self.entity_candidates_prior,
                                      self.prior_approx_mapping,
                                      self.num_entities,
                                      self.num_candidates,
                                      mention,
                                      label)
    candidates = get_candidate_strs(self.cursor, [self.entity_id_lookup[cand_id] for cand_id in candidate_ids.tolist()])
    return {'sentence_splits': self.sentence_splits[idx],
            'label': label,
            'embedded_page_content': self.embedded_documents[self.mention_doc_id[idx]],
            'entity_page_mentions': embed_page_content(self.embedding,
                                                       self.token_idx_lookup,
                                                       ' '.join(self.mentions_by_doc_id[self.mention_doc_id[idx]])),
            'p_prior': get_p_prior(self.entity_candidates_prior, self.prior_approx_mapping, mention, candidate_ids),
            'candidate_ids': candidate_ids,
            'candidate_mention_sim': torch.tensor([Levenshtein.ratio(mention, candidate)
                                                   for candidate in candidates])}

  def _getitem_sum_encoder(self, idx):
    idx = self.with_label[idx]
    label = self.entity_label_lookup.get(self.labels[idx], -1)
    mention = self.mentions[idx]
    candidate_ids = get_candidate_ids(self.entity_candidates_prior,
                                      self.prior_approx_mapping,
                                      self.num_entities,
                                      self.num_candidates,
                                      mention,
                                      label)
    candidates = get_candidate_strs(self.cursor, [self.entity_id_lookup[cand_id] for cand_id in candidate_ids.tolist()])
    return {'mention_sentence': self.mention_sentences[idx],
            'page_token_cnts': self.page_token_cnts_lookup[self.mention_doc_id[idx]],
            'label': label,
            'p_prior': get_p_prior(self.entity_candidates_prior, self.prior_approx_mapping, mention, candidate_ids),
            'candidate_ids': candidate_ids,
            'candidate_mention_sim': torch.tensor([Levenshtein.ratio(mention, candidate)
                                                   for candidate in candidates])}

  def _getitem_wiki2vec(self, idx):
    idx = self.with_label[idx]
    label = self.entity_label_lookup.get(self.labels[idx], -1)
    mention = self.mentions[idx]
    candidate_ids = get_candidate_ids(self.entity_candidates_prior,
                                      self.prior_approx_mapping,
                                      self.num_entities,
                                      self.num_candidates,
                                      mention,
                                      label)
    bag_of_nouns = get_bag_of_nouns(self.documents[self.mention_doc_id[idx]])
    candidates = get_candidate_strs(self.cursor, [self.entity_id_lookup[cand_id] for cand_id in candidate_ids.tolist()])
    return {'label': label,
            'bag_of_nouns': bag_of_nouns,
            'p_prior': get_p_prior(self.entity_candidates_prior, self.prior_approx_mapping, mention, candidate_ids),
            'candidate_ids': candidate_ids,
            'candidate_mention_sim': torch.tensor([Levenshtein.ratio(mention, candidate)
                                                   for candidate in candidates])}
