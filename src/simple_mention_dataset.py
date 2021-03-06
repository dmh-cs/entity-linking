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
from simple_dataset import SimpleDataset

class SimpleMentionDataset(SimpleDataset):
  def __init__(self,
               cursor,
               token_idx_lookup,
               full_token_idx_lookup,
               page_ids,
               lookups_path,
               idf_path,
               train_size,
               txt_dataset_path=None,
               pkl_dataset_prefix=None):
    super().__init__(cursor,
                     token_idx_lookup,
                     full_token_idx_lookup,
                     lookups_path,
                     idf_path,
                     train_size,
                     txt_dataset_path,
                     pkl_dataset_prefix=pkl_dataset_prefix)
    if txt_dataset_path is not None: return
    self.page_content_lim = 2000
    self.cursor = cursor
    self.page_ids = page_ids
    self.document_lookup = self.get_document_lookup(page_ids)
    self.mention_infos = self.get_mention_infos(page_ids)
    self.mentions = [info['mention'] for info in self.mention_infos]
    self.labels = [info['entity_id'] for info in self.mention_infos]
    self.mention_doc_id = [info['page_id'] for info in self.mention_infos]
    self.mention_sentences = get_mention_sentences_from_infos(self.document_lookup, self.mention_infos)
    self.mention_fs = [self._to_f(sentence) for sentence in self.mention_sentences]
    self.mention_fs_unstemmed = [self._to_f(sentence, stem_p=False)
                                 for sentence in self.mention_sentences]
    self.document_lookup = {page_id: parse_text_for_tokens(doc[:self.page_content_lim])
                            for page_id, doc in self.document_lookup.items()}
    self.page_f_lookup = {page_id: self._to_f(tokens)
                          for page_id, tokens in self.document_lookup.items()}
    self.page_f_lookup_unstemmed = {page_id: self._to_f(tokens, stem_p=False)
                                    for page_id, tokens in self.document_lookup.items()}
    self._post_init()

  def __len__(self):
    if self.txt_dataset_path is not None:
      return len(self.dataset_cache)
    else:
      return len(self.labels)

  def get_document_lookup(self, page_ids):
    self.cursor.execute(f'select id, content from pages where id in (' + str(page_ids)[1:-1] + ')')
    return {row['id']: row['content'] for row in self.cursor.fetchall()}

  def get_mention_infos(self, page_ids):
    self.cursor.execute('select mention, page_id, entity_id, mention_id, offset from entity_mentions_text where page_id in (' + str(page_ids)[1:-1] + ')')
    return self.cursor.fetchall()

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
  all_target_features = []
  all_candidate_features = []
  pair_ids = []
  features, candidate_ids, labels = zip(*batch)
  for mention_features, mention_candidate_ids, label in zip(features, candidate_ids, labels):
    target_idx = mention_candidate_ids.index(label)
    target_features = mention_features[target_idx]
    for candidate_features, candidate_id in zip(mention_features, mention_candidate_ids):
      if candidate_id != label:
        all_target_features.append(target_features)
        all_candidate_features.append(candidate_features)
        pair_ids.append((label, candidate_id))
  features = (torch.tensor(all_target_features), torch.tensor(all_candidate_features))
  pairwise_labels = torch.zeros(len(all_target_features), dtype=torch.float32)
  return features, pairwise_labels
