import pydash as _
import Levenshtein
from collections import Counter
import torch
from torch.utils.data import Dataset
import pickle
from nltk.stem.snowball import SnowballStemmer
import json

from data_fetchers import load_entity_candidate_ids_and_label_lookup, get_candidate_ids_simple, get_p_prior_cnts, get_candidate_strs
from parsers import parse_text_for_tokens
from data_transformers import get_mention_sentences_from_infos, pad_batch_list
import utils as u

def get_desc_fs(desc_fs, cursor, stemmer, cand_ids):
  def _process(page_content):
    tokenized = parse_text_for_tokens(page_content)
    return dict(Counter(stemmer.stem(token) for token in tokenized))
  cached_fs = {cand_id: desc_fs[cand_id] for cand_id in cand_ids if cand_id in desc_fs}
  remaining_cand_ids = [cand_id for cand_id in cand_ids if cand_id not in cached_fs]
  cursor.execute('select ep.entity_id as entity_id, left(p.content, 2000) as content from entity_by_page ep join pages p on ep.source_id = p.source_id where ep.entity_id in (' + str(remaining_cand_ids)[1:-1] + ')')
  return {row['entity_id']: _process(row['content']) for row in cursor.fetchall()}

class SimpleMentionDataset(Dataset):
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
    self.mention_sentences = get_mention_sentences_from_infos(self.document_lookup, self.mention_infos)
    self.mention_fs = [dict(Counter(self.stemmer.stem(token)
                                    for token in sentence))
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
    self.desc_fs = {}

  def __len__(self): return len(self.labels)

  def get_document_lookup(self, page_ids):
    self.cursor.execute(f'select id, content from pages where id in (' + str(page_ids)[1:-1] + ')')
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
    prior = get_p_prior_cnts(self.entity_candidates_prior,
                             self.prior_approx_mapping,
                             mention,
                             candidate_ids)
    times_mentioned = sum(prior)
    candidate_mention_sim = [Levenshtein.ratio(mention, candidate_str)
                             for candidate_str in candidate_strs]
    all_mentions_features = []
    candidate_fs = get_desc_fs(self.desc_fs, self.cursor, self.stemmer, candidate_ids)
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
                                    page_tfidf,
                                    candidate_mention_sim,
                                    candidate_prior,
                                    times_mentioned])
    return all_mentions_features, cands_with_page, label

def collate_simple_mention_pointwise(batch):
  pointwise_features = []
  pointwise_labels = []
  features, candidate_ids, labels = zip(*batch)
  for mention_features, mention_candidate_ids, label in zip(features, candidate_ids, labels):
    for candidate_features, candidate_id in zip(mention_features, mention_candidate_ids):
      pointwise_labels.append(1 if candidate_id == label else 0)
      pointwise_features.append(candidate_features)
  return torch.tensor(pointwise_features), torch.tensor(pointwise_labels, dtype=torch.float32)

def collate_simple_mention_pairwise(batch):
  pairwise_features = []
  pair_ids = []
  pairwise_labels = torch.zeros(len(batch), dtype=torch.float32)
  features, candidate_ids, labels = zip(*batch)
  for mention_features, mention_candidate_ids, label in zip(features, candidate_ids, labels):
    target_idx = mention_candidate_ids.index(label)
    target_features = features[target_idx]
    for candidate_features, candidate_id in zip(mention_features, mention_candidate_ids):
      if candidate_id != label:
        pairwise_features.append(target_features + candidate_features)
        pair_ids.append((label, candidate_id))
  return torch.tensor(pairwise_features), pairwise_labels
